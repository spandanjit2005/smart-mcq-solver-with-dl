import os, sys, torch, joblib, logging

import polars as pl

from pathlib import Path

project_root = str(Path(__file__).resolve().parents[3])

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from rag_pipeline.utils.config import get_logger, data_paths, model_config, data_vectorization_config
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer

from transformers import logging as hf_logging
from transformers.utils.logging import disable_progress_bar

CHUNKED_DATA_PARQUET = data_paths["parquet_chunk_path"]
EMBEDDED_DATA_JOBLIB = data_paths["embeddings_path"]

EMBEDDING_MODEL = model_config["embedder_model"]

EMBED_DIM = data_vectorization_config["embed_dim"]
BATCH_SIZE = data_vectorization_config["embed_batch_size"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger = get_logger("corpus_embedder")

# Configure HuggingFace API Key
load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_READ_TOKEN") if os.getenv("HF_READ_TOKEN") else "" # type: ignore

# Configure logging levels to hide model-loading report
hf_logging.set_verbosity_error()

disable_progress_bar()


chunk_data = pl.read_parquet(CHUNKED_DATA_PARQUET)

chunk_data = chunk_data.with_columns(
    pl.int_range(pl.len()).over(["source", "id"]).alias("chunk_idx")
).with_columns(
    (pl.col("source") + "_" + pl.col("id").cast(pl.Utf8) + "_chunk" + pl.col("chunk_idx").cast(pl.Utf8))
    .alias("chunk_id")
)

n_dupes = chunk_data.height - chunk_data["chunk_id"].n_unique()
assert n_dupes == 0, f"chunk_id collisions found: {n_dupes}; inspect duplicated (source, id) pairs"

texts = chunk_data["text"].to_list()
chunk_ids = chunk_data["chunk_id"].to_list()
orig_ids = chunk_data["id"].cast(pl.Utf8).to_list()
sources = chunk_data["source"].to_list()
titles = chunk_data["title"].to_list()
urls = chunk_data["url"].to_list()

n_chunks = chunk_data.height
logger.info(f"Loaded {n_chunks} chunks.")

logger.info(f"Loading {EMBEDDING_MODEL} for embedding chunks.")
model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE, model_kwargs={"dtype": torch.float16})

logger.info(f"Loaded {EMBEDDING_MODEL}, starting embedding.")

embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True
).astype("float32")
assert embeddings.shape == (n_chunks, EMBED_DIM), f"Unexpected shape: {embeddings.shape}"

logger.info(f"Embedded {n_chunks} across {n_chunks // BATCH_SIZE} batches.")

logger.info(f"Saving embedding data to {EMBEDDED_DATA_JOBLIB}")
joblib.dump({
    "embeddings": embeddings,   # numpy array shape (n_chunks, EMBED_DIM)
    "chunk_ids": chunk_ids,     # list of strings - globally unique
    "orig_ids": orig_ids,       # list of strings - raw per-source id
    "texts": texts,             # list of strings
    "sources": sources,         # list of strings
    "titles": titles,           # list of strings
    "urls": urls,               # list of strings
    "dim": EMBED_DIM            # embedding dimension
}, EMBEDDED_DATA_JOBLIB, compress=False)

logger.info("Successfully saved embedding data.")