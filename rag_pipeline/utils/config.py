import logging

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

# Define data paths for data ingestion and vectorization
data_paths = {
    "search_schedule_path": Path("rag_pipeline/data/search_schedules"),
    "dataset_path": Path("rag_pipeline/data/dataset"),
    "collection_log_path": Path("rag_pipeline/logs/data_collection"),

    "corpus_path": Path("rag_pipeline/data/data_corpus"),
    "vector_store_path": Path("rag_pipeline/data/vector_store"),

    "arxiv_records_path": Path("rag_pipeline/data/dataset/arxiv_records.parquet"),
    "pubmed_records_path": Path("rag_pipeline/data/dataset/pubmed_records.parquet"),
    "wiki_records_path": Path("rag_pipeline/data/dataset/wiki_records.parquet"),
    "curated_records_path": Path("rag_pipeline/data/dataset/curated_records.parquet"),

    "jsonl_chunk_path": Path("rag_pipeline/data/data_corpus/knowledge_chunks.jsonl"),
    "parquet_chunk_path":  Path("rag_pipeline/data/data_corpus/knowledge_chunks.parquet"),

    "embeddings_path": Path("rag_pipeline/data/vector_store/knowledge_embeddings.joblib"),
    "chroma_db_path": Path("rag_pipeline/data/vector_store/knowledge_db")
}

# Define models to use in the RAG Pipeline
model_config = {
    "chunker_model": "Qwen/Qwen3-Embedding-0.6B",
    "embedder_model": "Qwen/Qwen3-Embedding-4B",
    "reranker_model": "Qwen/Qwen3-Reranker-4B",
    "generative_slm": "Qwen/Qwen2.5-14B-Instruct"
}

# Define data ingestion config
data_ingestion_config = {
    "arxiv_delay_seconds": 3.05,
    "wiki_delay_seconds": 1.0,

    "TOTAL_RESULTS_PER_DOMAIN": 75,
    "MAX_QUERIES_PER_DOMAIN": 15,
    "MIN_RESULTS_PER_QUERY": 5,
    "MAX_RESULTS_HARD_CAP": 75
}

# Define data vectorization config
data_vectorization_config = {
    "embed_dim": 2560,
    "embed_batch_size": 16,

    "sentence_chunk_size": 512,
    "sentence_chunk_overlap": 128,

    "semantic_chunk_size": 768,
    "threshold": 0.7,
    "skip_window": 0, 
    "filter_window": 7,
    "similarity_window": 2
}

# Define chroma db config
chroma_db_config = {
    "chroma_db_name": "knowledge_db",

    "hnsw:space": "cosine",         
    "hnsw:M": 64,
    "hnsw:construction_ef": 256,
    "hnsw:search_ef": 128,
    "hnsw:batch_size": 128
}