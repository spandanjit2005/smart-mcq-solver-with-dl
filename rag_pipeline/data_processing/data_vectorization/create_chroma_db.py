import sys, joblib, chromadb

import numpy as np

from pathlib import Path

project_root = str(Path(__file__).resolve().parents[3])

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from rag_pipeline.utils.config import get_logger, data_paths, chroma_db_config


VECTOR_STORE_PATH = data_paths["vector_store_path"].item()

EMBEDDINGS_PATH = data_paths["embeddings_path"].item()
CHROMA_DB_PATH = data_paths["chroma_db_path"].item()

logger = get_logger("chroma_creator")

logger.info(f"Loading embeddings from {EMBEDDINGS_PATH}.")

data = joblib.load(EMBEDDINGS_PATH)

chunk_ids = data["chunk_ids"]
original_ids = data["orig_ids"]

embeddings = data["embeddings"]
texts = data["texts"]
titles = data["titles"]

sources = data["sources"]
urls = data["urls"]
dim = data["dim"]

n_vectors = embeddings.shape[0]
assert embeddings.shape == (n_vectors, dim), "Mismatch between embeddings and chunk metadata rows"
assert embeddings.dtype == np.float32, f"Embeddings must be float32 datatype, found embeddings of type: {embeddings.dtype}"

logger.info(f"Loaded {n_vectors} embeddings.")
logger.info(f"Creating Chroma DB persistent client.")

chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

collection = chroma_client.create_collection(
    name=chroma_db_config["chroma_db_name"].item(),
    metadata={
        "hnsw:space": chroma_db_config["hnsw:space"].item(),
        "hnsw:M": chroma_db_config["hnsw:M"].item(),

        "hnsw:construction_ef": chroma_db_config["hnsw:construction_ef"].item(),
        "hnsw:search_ef": chroma_db_config["hnsw:search_ef"].item(),

        "hnsw:batch_size": chroma_db_config["hnsw:batch_size"].item()
    }
)

chroma_metadata = [
    {
        "source": sources[i],
        "title": titles[i],
        "url": urls[i]
    }
    for i in range(len(embeddings))
]

batch_size = chroma_client.get_max_batch_size()
total_batches = (n_vectors + batch_size - 1) // batch_size

logger.info(f"Adding data to chroma db client in {total_batches} batches of batch size = {batch_size}")
for i in range(0, len(chunk_ids), batch_size):
    end_idx = min(i + batch_size, len(chunk_ids))

    collection.add(
        ids=chunk_ids[i:end_idx],
        embeddings=embeddings[i:end_idx].tolist(),
        documents=texts[i:end_idx],
        metadatas=chroma_metadata[i:end_idx] # type: ignore
    )
logger.info(f"Added data to chroma db.")