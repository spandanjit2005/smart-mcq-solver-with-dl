import sys, shutil, joblib, chromadb

import numpy as np

from pathlib import Path

project_root = str(Path(__file__).resolve().parents[3])

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from rag_pipeline.utils.config import get_logger, data_paths, data_vectorization_config, chroma_db_config


VECTOR_STORE_PATH = data_paths["vector_store_path"]
EMBEDDINGS_PATH = data_paths["embeddings_path"]
CHROMA_DB_PATH = data_paths["chroma_db_path"]

EMBED_DIM = data_vectorization_config["embed_dim"]

if CHROMA_DB_PATH.exists():
    shutil.rmtree(CHROMA_DB_PATH)

logger = get_logger("chroma_creator")

logger.info(f"Loading embeddings from {EMBEDDINGS_PATH}.")

data = joblib.load(EMBEDDINGS_PATH)

chunk_ids = data["chunk_id"]
original_ids = data["original_id"]
titles = data["title"]
embeddings = data["embedding"]
texts = data["text"]
sources = data["source"]
urls = data["url"]

n_vectors = embeddings.shape[0]
assert len(set(chunk_ids)) == len(chunk_ids), "Duplicate chunk_ids detected before insertion"
assert embeddings.shape == (n_vectors, EMBED_DIM), "Mismatch between embeddings and chunk metadata rows"
assert embeddings.dtype == np.float32, f"Embeddings must be float32 datatype, found embeddings of type: {embeddings.dtype}"

logger.info(f"Loaded {n_vectors} embeddings.")
logger.info(f"Creating Chroma DB persistent client.")

chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

collection = chroma_client.create_collection(
    name=chroma_db_config["chroma_db_name"],
    metadata={
        "hnsw:space": chroma_db_config["hnsw:space"],
        "hnsw:M": chroma_db_config["hnsw:M"],
        "hnsw:construction_ef": chroma_db_config["hnsw:construction_ef"],
        "hnsw:search_ef": chroma_db_config["hnsw:search_ef"],
        "hnsw:batch_size": chroma_db_config["hnsw:batch_size"]
    }
)

chroma_metadata = [
    {
        "title": titles[i],
        "source": sources[i],
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
    
logger.info("Insertion loop complete. Verifying by re-opening client from disk.")

del collection, chroma_client

verify_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
verify_collection = verify_client.get_collection(name=chroma_db_config["chroma_db_name"])

actual_count = verify_collection.count()
assert actual_count == n_vectors, f"Expected {n_vectors}, found {actual_count}"

all_embeddings_check = verify_collection.get(include=["embeddings"])
assert len(all_embeddings_check["ids"]) == n_vectors, \
    f"Expected {n_vectors} embeddings, got {len(all_embeddings_check['ids'])}"

embs = np.array(all_embeddings_check["embeddings"], dtype=np.float32)
norms = np.linalg.norm(embs, axis=1)
assert (norms > 0).all(), f"{(norms == 0).sum()} zero-norm embeddings found"
assert not np.isnan(embs).any(), "NaN values found in embeddings"

logger.info(f"Full verification passed: all {n_vectors} embeddings present and valid.")