import os, re, sys, logging

import polars as pl

from pathlib import Path

project_root = str(Path(__file__).resolve().parents[3])

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from rag_pipeline.utils.config import get_logger, data_paths, model_config, data_vectorization_config
from dotenv import load_dotenv

from transformers import AutoTokenizer

from chonkie import SentenceChunker, SemanticChunker

from transformers import logging as hf_logging
from transformers.utils.logging import disable_progress_bar

# Configure data paths
CORPUS_PATH = data_paths["corpus_path"]

logger = get_logger("corpus_chunker")

# Configure HuggingFace API Key
load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_READ_TOKEN") if os.getenv("HF_READ_TOKEN") else "" # type: ignore

# Configure logging levels to hide model-loading report
hf_logging.set_verbosity_error()

disable_progress_bar()

tokenizer = AutoTokenizer.from_pretrained(model_config["embedder_model"])

logger.info(f"Loading chunkers from chonkie.")

# Initialize SentenceChunker for arXiv/PubMed (Academic abstracts)
sentence_chunker = SentenceChunker(
    tokenizer=tokenizer,
    chunk_size=data_vectorization_config["sentence_chunk_size"],
    chunk_overlap=data_vectorization_config["sentence_chunk_overlap"]
)

# Initialize SemanticChunker for Wikipedia (Long-form articles)
semantic_chunker = SemanticChunker(
    embedding_model=model_config["chunker_model"],
    chunk_size=data_vectorization_config["semantic_chunk_size"],
    threshold=data_vectorization_config["threshold"],
    skip_window=data_vectorization_config["skip_window"], 
    filter_window=data_vectorization_config["filter_window"],
    similarity_window=data_vectorization_config["similarity_window"]
)

# Define function to clean Wikipedia sections and delete math markers
SECTION_STOPWORDS = {
    "references", "further reading", "external links", "see also",
    "notes", "bibliography", "citations", "sources", "footnotes",
    "gallery", "in popular culture",
}

TEX_START = re.compile(r"\{\\(displaystyle|textstyle|scriptstyle|scriptscriptstyle)\b")
GARBLED_PREFIX = re.compile(r"(?:\S+\s{2,}){2,}\S*\s*$")

def _find_balanced_end(text: str, start: int):
    """text[start] == '{'; returns index just past the matching '}'."""
    depth = 0

    for i in range(start, len(text)):

        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1

            if depth == 0:
                return i + 1
            
    return len(text)

def _strip_math(text: str) -> str:
    """Removes {\\displaystyle ...} TeX blocks and garbled
    Unicode-spaced rendering that precedes them."""
    out = []
    i = 0

    while i < len(text):
        m = TEX_START.match(text, i)

        if m:
            if out:
                prefix = "".join(out)
                gm = GARBLED_PREFIX.search(prefix)

                if gm:
                    out = [prefix[:gm.start()]]

            i = _find_balanced_end(text, m.start())

        else:
            out.append(text[i])
            i += 1

    return "".join(out)

def _truncate_at_boilerplate(text: str) -> str:
    """Cuts everything from the first boilerplate section title onward."""
    lines = text.split("\n")\
    
    for i, line in enumerate(lines):
        if line.strip().lower() in SECTION_STOPWORDS:
            return "\n".join(lines[:i])
        
    return text

def clean_wikipedia_text(text: str) -> str:
    if not text:
        return ""
    
    text = _truncate_at_boilerplate(text)

    text = _strip_math(text)

    # Citation markers: [1], [23], [citation needed]
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(
        r"\[(citation needed|clarification needed|when\?|who\?)\]",
        "", text, flags=re.IGNORECASE,
    )

    # Leftover table/image/link artifacts
    text = re.sub(r"\{\|.*?\|\}", "", text, flags=re.DOTALL)           # wiki tables
    text = re.sub(r"\[\[(File|Image):.*?\]\]", "", text,
                   flags=re.IGNORECASE | re.DOTALL)                    # image embeds
    text = re.sub(r"\[\[[^\]|]*\|([^\]]+)\]\]", r"\1", text)           # [[target|display]]
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)                    # [[target]]
    text = re.sub(r"https?://\S+", "", text)                           # bare URLs

    # Whitespace cleanup
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# Start chunking data
data_chunks = []

logger.info(f"Processing arXiv data.")
arxiv_data = pl.read_parquet(data_paths["arxiv_records_path"])

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
pubmed_data = pl.read_parquet(data_paths["pubmed_records_path"])

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
wiki_data = pl.read_parquet(data_paths["wiki_records_path"])

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
curated_data = pl.read_parquet(data_paths["curated_records_path"])

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
data_chunks_df.write_ndjson(data_paths["jsonl_chunk_path"])
data_chunks_df.write_parquet(data_paths["parquet_chunk_path"], compression="zstd")

logger.info(f"Saved {len(data_chunks)} chunks to disk at {CORPUS_PATH}.")