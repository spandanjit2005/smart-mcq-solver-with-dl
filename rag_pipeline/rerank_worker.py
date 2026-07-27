# Define script to parallelize ranking of retrieved documents
import gc, sys, torch, argparse

from tqdm.auto import tqdm

import numpy as np
import polars as pl

from pathlib import Path

from sentence_transformers import CrossEncoder
from transformers import BitsAndBytesConfig

def load_reranker(local_path: Path, hf_id: str, bnb_config: BitsAndBytesConfig, args):
    path_str = str(local_path)
    is_saved = (local_path / "config.json").exists() 
    model_name_or_path = path_str if is_saved else hf_id

    print(f"{'Loading' if is_saved else 'Downloading'} Reranker: {model_name_or_path}")

    model = CrossEncoder(
        model_name_or_path,
        trust_remote_code=True,
        model_kwargs={
            "quantization_config": bnb_config,
            "dtype": torch.float16,
            "device_map": "cuda:0"
        },
        prompts={"rerank": args.instruction},
        default_prompt_name="rerank"
    )

    if not is_saved:
        model.save(path_str)

    return model

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--local_model_path", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--instruction", required=True)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--k", type=int, default=5)
    args = p.parse_args()

    payload = pl.read_parquet(args.input)
    query_texts = payload["query"].to_list()
    retrieved_chunks = payload["retrieved_chunks"].to_list()

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False
    )

    model = load_reranker(
        local_path=Path(args.local_model_path),
        hf_id=args.model_path,
        bnb_config=bnb_config,
        args=args
    )

    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token

    model.model.config.pad_token_id = model.tokenizer.pad_token_id # type: ignore
    model.model.config.use_cache = False # type: ignore
    model.model.eval() # type: ignore

    flat_pairs, boundaries = [], []
    idx = 0

    for q, chunks in zip(query_texts, retrieved_chunks):
        chunks = [c if c.strip() else " " for c in chunks]
        flat_pairs.extend((q, c) for c in chunks)
        boundaries.append((idx, idx + len(chunks)))
        idx += len(chunks)

    flat_scores = model.predict(
        flat_pairs,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    top_chunks_per_query = []
    for (start, end), chunks in zip(boundaries, retrieved_chunks):
        scores = flat_scores[start:end]
        top_idx = np.argsort(-scores)[:args.k]
        top_chunks_per_query.append([chunks[i] for i in top_idx])

    pl.DataFrame({"top_chunks": top_chunks_per_query}).write_parquet(args.output)

if __name__ == "__main__":
    main()
