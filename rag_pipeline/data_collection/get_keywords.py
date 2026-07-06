import os, gc, json, logging

import polars as pl

from tqdm import tqdm
from dotenv import load_dotenv

import torch

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from transformers import logging as hf_logging
from transformers.utils.logging import disable_progress_bar

TRAIN_DATA_PATH = "data/train.csv"

train_data = pl.read_csv(TRAIN_DATA_PATH)

# Configure text columns
OPTION_COLS = ['prompt', 'A', 'B', 'C', 'D', 'E']

# Configue Domain Classifier model
DOMAIN_CLASSIFIER_LM = "google/gemma-4-12B-it"

# Configure Hugging Face token for faster downloads
load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_READ_TOKEN") or ""

# Configure logging levels to hide model-loading report
hf_logging.set_verbosity_error()

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

disable_progress_bar()

train_data = train_data.with_columns(
    pl.concat_str([pl.col(c).fill_null("") for c in OPTION_COLS], separator=" ")
    .str.slice(0, 1024)
    .alias("truncated_text")
)

texts_to_process = train_data["truncated_text"].to_list()

domain_chat_inputs = [
    [
        {
            "role": "system", 
            "content": (
                "You are an expert Data Taxonomist and RAG Architecture Consultant. "
                "Analyze the provided text to identify its specific academic, technical, or professional domain. "
                "You must provide exactly two things:\n"
                "1. Domain: A concise label (1-3 words) with high semantic separation.\n"
                "2. Keywords: 3 to 5 highly specific, technical keywords or keyphrases extracted from the text, separated strictly by semicolons (;).\n\n"
                "Constraints: Do not output numbers. Do not output multiple-choice options. Respond strictly in the exact format shown below."
            )
        },
        # Example 1: Biology 
        {
            "role": "user", 
            "content": (
                "Text: Which of the following is true about mitochondria? "
                "A) It is the powerhouse of the cell "
                "B) Found in plants only "
                "C) It is a virus "
                "D) None of the above "
                "E) All of the above"
            )
        },
        {
            "role": "assistant", 
            "content": (
                "Domain: Cell Biology\n"
                "Keywords: mitochondria; cellular organelles; powerhouse of the cell; bioenergetics"
            )
        },
        # Example 2: Physics
        {
            "role": "user", 
            "content": (
                "Text: Calculate the terminal velocity of a 10kg mass dropping in a vacuum. "
                "A) 9.8m/s "
                "B) 0m/s "
                "C) 98m/s "
                "D) Infinite "
                "E) 10m/s"
            )
        },
        {
            "role": "assistant", 
            "content": (
                "Domain: Classical Mechanics\n"
                "Keywords: terminal velocity; kinematics; Newtonian physics; vacuum acceleration"
            )
        },
        # The Actual Data
        {
            "role": "user", 
            "content": f"Text: {text}"
        }
    ]
    for text in texts_to_process
]

tokenizer = AutoTokenizer.from_pretrained(DOMAIN_CLASSIFIER_LM, padding_side="left")

model = AutoModelForCausalLM.from_pretrained(
    DOMAIN_CLASSIFIER_LM,
    device_map="auto",
    trust_remote_code=True,
    dtype=torch.float16
)

def process_batches(chat_inputs, batch_size=8):
    all_results = []

    torch.cuda.empty_cache()
    gc.collect()
    
    for i in tqdm(range(0, len(chat_inputs), batch_size)):
        batch = chat_inputs[i : i + batch_size]
        
        encoded_batch = tokenizer.apply_chat_template(
            batch,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_dict=True
        ).to(model.device)

        with torch.inference_mode():
            output_tokens = model.generate( # type: ignore
                **encoded_batch, 
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        
        input_len = encoded_batch["input_ids"].shape[1]
        new_tokens = output_tokens[:, input_len:]
        
        decoded_batch = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        all_results.extend(decoded_batch)

        del encoded_batch, output_tokens, new_tokens
        torch.cuda.empty_cache()        
        
    return all_results

extracted_text = process_batches(domain_chat_inputs, batch_size=8)

def merge_similar_keywords(keyword_list):
    if keyword_list is None or len(keyword_list) == 0:
        return []
        
    standard_list = list(keyword_list)
        
    sets = [set(lst) for lst in standard_list if lst is not None and len(lst) > 0]
    
    merged = True
    while merged:
        merged = False
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                s1, s2 = sets[i], sets[j]
                if not s1 or not s2:
                    continue
                    
                intersection = len(s1 & s2)
                union = len(s1 | s2)
                
                if union > 0 and (intersection / union) > 0.5:
                    sets[j] = s1 | s2
                    sets[i] = set()
                    merged = True
                    break
            
            if merged:
                break
        
        sets = [s for s in sets if s]
        
    return [list(s) for s in sets]

train_data = train_data.with_columns(pl.Series(name="raw_output", values=extracted_text))

train_data = train_data.with_columns(
    pl.col("raw_output")
    .str.extract(r"Domain:\s*(.*?)(?:\n|$)", 1)
    .str.strip_chars()
    .str.to_titlecase()
    .alias("domain_name"),
    
    pl.col("raw_output")
    .str.extract(r"Keywords:\s*(.*?)(?:\n|$)", 1)
    .str.strip_chars()
    .alias("domain_keywords")
)

keywords = train_data.with_columns(
    pl.col("domain_keywords")
    .fill_null("")
    .str.split(";")
    .list.eval(pl.element().str.strip_chars().filter(pl.element() != ""))
    .alias("keyword_list")
)

domain_counts = (
    keywords
    .group_by("domain_name")
    .agg(
        pl.len().alias("count"),
        pl.col("keyword_list").alias("grouped_keywords")
    )
    .with_columns(
        pl.col("grouped_keywords")
        .map_elements(merge_similar_keywords, return_dtype=pl.List(pl.List(pl.String)))
        .alias("domain_keywords")
    )
)

domain_counts = domain_counts.filter(pl.col("count") >= 5).sort("count", descending=True)
total_frequent_unique = domain_counts.height

train_data.select(
    [
        "id",
        "prompt", 
        pl.coalesce([
            pl.when(pl.col("answer") == opt).then(pl.col(opt)) 
            for opt in OPTION_COLS
        ]).alias("correct_answer_text"),
        "domain_name", 
        "domain_keywords"
    ]
).write_csv("rag_pipeline/data/domains/train_data_keyword.csv")

domain_counts.with_columns(
    pl.col("domain_keywords")
    .map_elements(
        lambda x: json.dumps(x.to_list()) if x is not None else None, 
        return_dtype=pl.String
        )
    ).drop("grouped_keywords").write_csv("rag_pipeline/data/domains/domain_count_keyword.csv")

print(f"Total number of unique topic labels: {train_data['domain_name'].n_unique()}")
print(f"Total number of unique topic labels with a count >= 5: {total_frequent_unique}")

total_sample_count = 0

for row in domain_counts.iter_rows(named=True):
    total_sample_count += row["count"]
    print(f"Count: {row["count"]} | Label: {row["domain_name"]}")

print(f"\nCount of samples covered by {total_frequent_unique} domains: {total_sample_count}")