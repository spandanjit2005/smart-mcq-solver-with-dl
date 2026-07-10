import torch, joblib, logging

import polars as pl

from pathlib import Path
from datetime import datetime, timezone, timedelta

from sentence_transformers import SentenceTransformer

from transformers import logging as hf_logging
from transformers.utils.logging import disable_progress_bar

CHUNKED_DATA_PARQUET = Path("rag_pipeline/data/vector_store/knowledge_chunks.parquet")
EMBEDDED_DATA_JOBLIB = Path("rag_pipeline/data/vector_store/knowledge_embeddings.joblib")

EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-4B"

EMBED_DIM = 2560
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Configure custom logger
IST = timezone(timedelta(hours=5, minutes=30))

logging.Formatter.converter = staticmethod(
    lambda ts: datetime.fromtimestamp(ts, tz=IST).timetuple()
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

logger = logging.getLogger("corpus_embedder")

# Configure logging levels to hide model-loading report
hf_logging.set_verbosity_error()

logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

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