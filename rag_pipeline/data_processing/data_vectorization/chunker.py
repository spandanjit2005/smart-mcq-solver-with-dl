import re, logging

import polars as pl

from pathlib import Path
from datetime import datetime, timezone, timedelta

from transformers import AutoTokenizer

from chonkie import SentenceChunker, SemanticChunker

from transformers import logging as hf_logging
from transformers.utils.logging import disable_progress_bar

# Configure models and data paths
CHUNK_MODEL = "Qwen/Qwen3-Embedding-0.6B"
VECTOR_MODEL = "Qwen/Qwen3-Embedding-4B"

DATASET_PATH = Path("rag_pipeline/data/dataset")
CORPUS_PATH = Path("rag_pipeline/data/data_corpus")

# ARXIV_DATASET = DATASET_PATH / "arxiv_records.parquet"
# PUBMED_DATASET = DATASET_PATH / "pubmed_records.parquet"
# WIKIPEDIA_DATASET = DATASET_PATH / "wiki_records.parquet"
# CURATED_DATASET = DATASET_PATH / "curated_records.parquet"

# CHUNKED_DATA_JSONL = CORPUS_PATH / "knowledge_chunks.jsonl"
# CHUNKED_DATA_PARQUET = CORPUS_PATH / "knowledge_chunks.parquet"

# Configure logger
IST = timezone(timedelta(hours=5, minutes=30))

logging.Formatter.converter = staticmethod(
    lambda ts: datetime.fromtimestamp(ts, tz=IST).timetuple()
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("corpus_chunker")

# Configure logging levels to hide model-loading report
hf_logging.set_verbosity_error()

logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

disable_progress_bar()

tokenizer = AutoTokenizer.from_pretrained(VECTOR_MODEL)

logger.info(f"Loading chunkers from chonkie.")

# Initialize SentenceChunker for arXiv/PubMed (Academic abstracts)
sentence_chunker = SentenceChunker(
    tokenizer=tokenizer,
    chunk_size=512,
    chunk_overlap=128
)

# Initialize SemanticChunker for Wikipedia (Long-form articles)
semantic_chunker = SemanticChunker(
    embedding_model=CHUNK_MODEL,
    chunk_size=768,
    threshold=0.7,
    skip_window=0, 
    filter_window=7,
    similarity_window=2
)

# Define function to clean Wikipedia sections and delete math markers
def clean_wikipedia_text(
    text: str,
    section_headers=("References", "Further reading", "External links", "See also"),
    math_markers=(r'{\displaystyle', r'{\textstyle')
) -> str:
    
    if not text:
        return ""
    
    pattern = r'\n(' + '|'.join(re.escape(h) for h in section_headers) + r')\n'
    match = re.search(pattern, text)
    
    if match:
        text = text[:match.start()]

    out = []
    i = 0
    while i < len(text):
        candidates = [(text.find(m, i), m) for m in math_markers]
        candidates = [(idx, m) for idx, m in candidates if idx != -1]
        
        if not candidates:
            out.append(text[i:])
            break
            
        idx, marker = min(candidates, key=lambda c: c[0])

        depth = 0
        j = idx
        while j < len(text):
            if text[j] == '{':
                depth += 1
                
            elif text[j] == '}':
                depth -= 1
                
                if depth == 0:
                    j += 1
                    break
            j += 1
        block_end = j

        k = idx
        last_break = None
        
        while k > 0:
            line_start = text.rfind('\n', 0, k)
            line = text[line_start + 1:k]
            stripped = line.strip()
            
            if stripped == '':
                last_break = line_start + 1
                k = line_start
                continue
                
            if re.search(r'[A-Za-z]{4,}', stripped):
                break
                
            last_break = line_start
            k = line_start
            
        linearized_start = last_break if last_break is not None else idx

        out.append(text[i:linearized_start])
        i = block_end

    text = ''.join(out)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()

# Start chunking data
data_chunks = []

logger.info(f"Processing arXiv data.")
arxiv_data = pl.read_parquet(DATASET_PATH / "arxiv_records.parquet")

for row in arxiv_data.iter_rows(named=True):
    text_to_chunk = f"{row["title"]}. {row["abstract"]}"
    chunks = sentence_chunker(text_to_chunk)
    
    for chunk in chunks:
        arxiv_chunk = {
            "text": chunk.text, # type: ignore
            "source": "arxiv",
            "id": row["id"],
            "title": row["title"],
            "url": row["url"]
        }
        
        data_chunks.append(arxiv_chunk)
logger.info(f"Finished processing arXiv data.")

logger.info(f"Processing PubMed data.")
pubmed_data = pl.read_parquet(DATASET_PATH / "pubmed_records.parquet")

for row in pubmed_data.iter_rows(named=True):
    text_to_chunk = f"{row["title"]}. {row["abstract"]}"
    chunks = sentence_chunker(text_to_chunk)
    
    for chunk in chunks:
        pubmed_chunk = {
            "text": chunk.text, # type: ignore
            "source": "pubmed",
            "id": row["id"],
            "title": row["title"],
            "url": row["url"]
        }

        data_chunks.append(pubmed_chunk)
logger.info(f"Finished processing PubMed data.")

logger.info(f"Processing Wikipedia data.")
wiki_data = pl.read_parquet(DATASET_PATH / "wiki_records.parquet")

for row in wiki_data.iter_rows(named=True):
    clean_full_text = clean_wikipedia_text(row["full_text"])
    text_to_chunk = f"{row['title']}. {row['abstract']} {clean_full_text}"
    chunks = semantic_chunker(text_to_chunk)
    
    for chunk in chunks:
        wiki_chunk = {
            "text": chunk.text, # type: ignore
            "source": "wikipedia",
            "id": row["id"],
            "title": row["title"],
            "url": row["url"]
        }

        data_chunks.append(wiki_chunk)
logger.info(f"Finished processing Wikipedia data.")

logger.info(f"Processing curated data.")
curated_data = pl.read_parquet(DATASET_PATH / "curated_records.parquet")

for row in curated_data.iter_rows(named=True):
    clean_full_text = clean_wikipedia_text(row["full_text"])
    text_to_chunk = f"{row['title']}. {row['abstract']} {clean_full_text}"
    chunks = semantic_chunker(text_to_chunk)
    
    for chunk in chunks:
        curated_chunk = {
            "text": chunk.text, # type: ignore
            "source": "curated",
            "id": row["id"],
            "title": row["title"],
            "url": row["url"]
        }

        data_chunks.append(curated_chunk)
logger.info(f"Finished processing curated data.")


logger.info(f"Total chunks created: {len(data_chunks)}")

# Save chunked data as both jsonl and parquet
data_chunks_df = pl.DataFrame(data_chunks)
data_chunks_df.write_ndjson(CORPUS_PATH / "knowledge_chunks.jsonl")
data_chunks_df.write_parquet(CORPUS_PATH / "knowledge_chunks.parquet", compression="zstd")

logger.info(f"Saved {len(data_chunks)} chunks to disk at {CORPUS_PATH}.")