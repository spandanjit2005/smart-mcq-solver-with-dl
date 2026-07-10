import logging

import polars as pl

from pathlib import Path
from datetime import datetime, timezone, timedelta

IST = timezone(timedelta(hours=5, minutes=30))

# Define Logger config
logging.Formatter.converter = staticmethod(
    lambda ts: datetime.fromtimestamp(ts, tz=IST).timetuple()
)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logging.getLogger("arxiv").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("wikipediaapi").setLevel(logging.WARNING)

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"{name}")

# Define data ingestion config
data_ingestion_config = pl.DataFrame({
    "arxiv_delay_seconds": 3.05,
    "wiki_delay_seconds": 1.0,

    "TOTAL_RESULTS_PER_DOMAIN": 75,
    "MAX_QUERIES_PER_DOMAIN": 15,
    "MIN_RESULTS_PER_QUERY": 5,
    "MAX_RESULTS_HARD_CAP": 75
})

# Define data vectorization config
data_vectorization_config = pl.DataFrame({
    "chunker_model": "Qwen/Qwen3-Embedding-0.6B",
    "embedder_model": "Qwen/Qwen3-Embedding-4B",
    "embed_dim": 2560,
    "embed_batch_size": 32,

    "sentence_chunk_size": 512,
    "sentence_chunk_overlap": 128,

    "semantic_chunk_size": 768,
    "threshold": 0.7,
    "skip_window": 0, 
    "filter_window": 7,
    "similarity_window": 2,

    "dataset_path": Path("rag_pipeline/data/dataset"),
    "corpus_path": Path("rag_pipeline/data/data_corpus"),
    "vector_store_path": Path("rag_pipeline/data/vector_store")
})