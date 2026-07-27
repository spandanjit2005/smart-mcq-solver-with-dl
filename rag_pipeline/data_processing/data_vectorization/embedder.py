import os, sys, torch, joblib

import numpy as np
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
CURATED_DATA_PARQUET = data_paths["curated_data_path"]
EMBEDDED_DATA_JOBLIB = data_paths["embeddings_path"]

EMBEDDING_MODEL = model_config["embedder_model"]

EMBED_DIM = data_vectorization_config["embed_dim"]
BATCH_SIZE = data_vectorization_config["embed_batch_size"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

COLS = ["chunk_id", "id", "title", "text", "source", "url"]

logger = get_logger("corpus_embedder")

# Configure HuggingFace API Key
load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_READ_TOKEN") if os.getenv("HF_READ_TOKEN") else "" # type: ignore

# Configure logging levels to hide model-loading report
hf_logging.set_verbosity_error()

disable_progress_bar()

# Load chunked data
chunk_data = pl.read_parquet(CHUNKED_DATA_PARQUET)

chunk_data = chunk_data.with_columns(
    pl.int_range(pl.len()).over(["source", "id"]).alias("chunk_idx")
).with_columns(
    (pl.col("source") + "_" + pl.col("id").cast(pl.Utf8) + "_chunk" + pl.col("chunk_idx").cast(pl.Utf8))
    .alias("chunk_id")
).drop("chunk_idx")

n_dupes = chunk_data.height - chunk_data["chunk_id"].n_unique()
assert n_dupes == 0, f"chunk_id collisions found: {n_dupes}; inspect duplicated (source, id) pairs"

chunk_texts = chunk_data["text"].to_list()
n_chunks = chunk_data.height
logger.info(f"Loaded {n_chunks} chunks.")

# Load curated data
curated_data = pl.read_parquet(CURATED_DATA_PARQUET)

curated_texts = curated_data["text"].to_list()
n_curated = curated_data.height
logger.info(f"Loaded {n_curated} curated data samples.")

# Load embedding model and encode data
logger.info(f"Loading {EMBEDDING_MODEL} for embedding chunks.")
model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE, model_kwargs={"dtype": torch.float16})

logger.info(f"Loaded {EMBEDDING_MODEL}, starting embedding.")

chunk_embeddings = model.encode(
    chunk_texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True
).astype("float32")
assert chunk_embeddings.shape == (n_chunks, EMBED_DIM), f"Unexpected shape for chunk embeddings: {chunk_embeddings.shape}"

curated_embeddings = model.encode(
    curated_texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True
).astype("float32")
assert curated_embeddings.shape == (n_curated, EMBED_DIM), f"Unexpected shape for curated embeddings: {curated_embeddings.shape}"

logger.info(f"Embedded {n_chunks + n_curated} across {(n_chunks + n_curated) // BATCH_SIZE} batches.")

# Save embedding data
logger.info(f"Saving embedding data to {EMBEDDED_DATA_JOBLIB}")
EMBEDDED_DATA_JOBLIB.parent.mkdir(parents=True, exist_ok=True)

combined_data = pl.concat(
    [chunk_data.select(COLS), curated_data.select(COLS)],
    how="vertical"
)
combined_texts = chunk_texts + curated_texts
combined_embeddings = np.concatenate([chunk_embeddings, curated_embeddings], axis=0)

joblib.dump({
    "chunk_id": combined_data["chunk_id"].to_list(),  # list of strings - globally unique
    "original_id": combined_data["id"].to_list(),     # list of strings - raw per-source id
    "title": combined_data["title"].to_list(),        # list of strings
    "text": combined_texts,                           # list of strings
    "embedding": combined_embeddings,                 # numpy array of shape (n_chunks, EMBED_DIM)
    "source": combined_data["source"].to_list(),      # list of strings
    "url": combined_data["url"].to_list()             # list of strings
}, EMBEDDED_DATA_JOBLIB, compress=False)

logger.info("Successfully saved embedding data.")