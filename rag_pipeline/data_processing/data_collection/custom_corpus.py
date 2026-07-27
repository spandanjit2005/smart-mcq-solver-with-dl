import sys, zipfile, subprocess

from pathlib import Path

project_root = str(Path(__file__).resolve().parents[3])

if project_root not in sys.path:
    sys.path.insert(0, project_root)

import polars as pl

from rag_pipeline.utils.config import data_paths

HF_URL = "https://huggingface.co/datasets/Sangeetha/Kaggle-LLM-Science-Exam/resolve/main/train_final_LLMScience.csv"
KAGGLE_URL = "https://www.kaggle.com/competitions/kaggle-llm-science-exam/data"
KAGGLE_COMPETITION = "kaggle-llm-science-exam"

DATASET_PATH = data_paths["dataset_path"]
DATASET_PATH.mkdir(exist_ok=True)

CORPUS_PATH = data_paths["corpus_path"]

TITLE = "Kaggle LLM Science Exam"
OPTION_COLS = ["A", "B", "C", "D", "E"]

# Define functions to get Huggingface and Kaggle Datasets
def get_hf_data() -> Path:
    dest = DATASET_PATH / "huggingface_train.csv"
    if not dest.exists():
        df = pl.read_csv(HF_URL)
        df.write_csv(dest)

    return dest

def get_kaggle_data() -> Path:
    dest = DATASET_PATH / "train.csv"
    if not dest.exists():
        subprocess.run(
            [
                "kaggle", "competitions", "download",
                "-c", KAGGLE_COMPETITION,
                "-f", "train.csv",
                "-p", str(DATASET_PATH),
            ],
            check=True,
        )
        zip_path = DATASET_PATH / "train.csv.zip"

        if zip_path.exists():
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(DATASET_PATH)

            zip_path.unlink()

    return dest

# Define function to extract text by concatenating `prompt` and answer text
def extract_text() -> pl.Expr:
    expr = pl.lit(None, dtype=pl.String)

    for letter in OPTION_COLS:
        expr = pl.when(pl.col("answer") == letter).then(pl.col(letter)).otherwise(expr)

    return expr

def get_data(data: pl.DataFrame, source: str, url: str) -> pl.DataFrame:
    return (
        data.with_row_index(name="row_idx")
        .with_columns(
            pl.lit(TITLE).alias("title"),
            pl.lit(url).alias("url"),
            pl.lit(source).alias("source"),
            pl.lit(source).alias("id"),
            extract_text().alias("answer_text"),
        )
        .with_columns(
            (pl.col("id") + "_" + pl.col("row_idx").cast(pl.String)).alias("chunk_id"),
            pl.concat_str(
                [
                    pl.lit("Prompt: "),
                    pl.col("prompt"),
                    pl.lit(" Answer: "),
                    pl.col("answer_text"),
                ]
            ).alias("text"),
        )
        .select(["text", "chunk_id", "id", "source", "title", "url"])
    )


def main():
    hf_path = get_hf_data()
    kaggle_path = get_kaggle_data()

    hf_data = pl.read_csv(hf_path)
    kaggle_data = pl.read_csv(kaggle_path)

    hf_corpus = get_data(hf_data, source="huggingface", url="https://huggingface.co/datasets/Sangeetha/Kaggle-LLM-Science-Exam")
    kaggle_corpus = get_data(kaggle_data, source="kaggle", url=KAGGLE_URL)

    corpus = pl.concat([hf_corpus, kaggle_corpus], how="vertical")

    out_path = data_paths["curated_data_path"]
    corpus.write_parquet(out_path)
    print(f"Saved {corpus.height} rows to {out_path}.")

if __name__ == "__main__":
    main()