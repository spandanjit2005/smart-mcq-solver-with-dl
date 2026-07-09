import os, time, signal
import concurrent.futures

import polars as pl

from pathlib import Path

from utils.core_utils import get_logger

from utils.arxiv_utils import fetch_arxiv_data
from utils.pubmed_utils import fetch_pubmed_data
from utils.wiki_utils import fetch_wiki_data
from utils.curated_utils import fetch_curated_data

SEARCH_SCHEDULE_DIR = Path("rag_pipeline/data/search_schedules")
OUTPUT_DATASET_DIR = Path("rag_pipeline/data/dataset")
OUTPUT_LOG_DIR = Path("rag_pipeline/logs/data_collection")

data_logger = get_logger("all_data")

config_data = pl.DataFrame({
    "TOTAL_RESULTS_PER_DOMAIN": [75],
    "MAX_QUERIES_PER_DOMAIN": [15],
    "MIN_RESULTS_PER_QUERY": [5],
    "MAX_RESULTS_HARD_CAP": [75]
})

def force_kill_handler(signum, frame):
    data_logger.warning("KeyboardInterrupt. Forcefully shutting down parallel data ingestion.")
    os._exit(1)

def parallel_ingestion(
    config_data: pl.DataFrame,
    INPUT_CSV: Path,
    OUTPUT_DATASET_DIR: Path,
    OUTPUT_LOG_DIR: Path,
) -> None:
    
    OUTPUT_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_LOG_DIR.mkdir(parents=True, exist_ok=True)

    signal.signal(signal.SIGINT, force_kill_handler)

    data_logger.info("Starting parallel data ingestion.")
    start_time = time.perf_counter()

    # fetch_arxiv_data(
    #     config_data = config_data,
    #     INPUT_CSV = INPUT_CSV / "arxiv_search_schedule.csv",
    #     OUTPUT_RECORDS = OUTPUT_DATASET_DIR / "arxiv_dataset.parquet",
    #     OUTPUT_LOGS = OUTPUT_LOG_DIR / "arxiv_ingestion_logs.csv"
    # )

    # fetch_pubmed_data(
    #     config_data = config_data,
    #     INPUT_CSV = INPUT_CSV / "pubmed_search_schedule.csv",
    #     OUTPUT_RECORDS = OUTPUT_DATASET_DIR / "pubmed_dataset.parquet",
    #     OUTPUT_LOGS = OUTPUT_LOG_DIR / "pubmed_ingestion_logs.csv"        
    # )

    # fetch_wiki_data(
    #     config_data = config_data,
    #     INPUT_CSV = INPUT_CSV / "wiki_search_schedule.csv",
    #     CURATED_CSV = CURATED_CSV,
    #     OUTPUT_RECORDS = OUTPUT_DATASET_DIR / "wiki_dataset.parquet",
    #     OUTPUT_LOGS = OUTPUT_LOG_DIR / "wiki_ingestion_logs.csv"
    # )

    # fetch_curated_data(
    #     INPUT_CSV = INPUT_CSV / "curated_search_schedule.csv",
    #     OUTPUT_RECORDS = OUTPUT_DATASET_DIR / "curated_dataset.parquet",
    #     OUTPUT_LOGS = OUTPUT_LOG_DIR / "curated_ingestion_logs.csv"
    # )

    tasks = {
        "arxiv": {
            "func": fetch_arxiv_data,
            "kwargs": {
                "config_data": config_data,
                "INPUT_CSV": INPUT_CSV / "arxiv_search_schedule.csv",
                "OUTPUT_RECORDS": OUTPUT_DATASET_DIR / "arxiv_records.parquet",
                "OUTPUT_LOGS": OUTPUT_LOG_DIR / "arxiv_logs.csv"
            }
        },
        "wiki": {
            "func": fetch_wiki_data,
            "kwargs": {
                "config_data": config_data,
                "INPUT_CSV": INPUT_CSV / "wiki_search_schedule.csv",
                "CURATED_CSV": INPUT_CSV / "curated_search_schedule.csv",
                "OUTPUT_RECORDS": OUTPUT_DATASET_DIR / "wiki_records.parquet",
                "OUTPUT_LOGS": OUTPUT_LOG_DIR / "wiki_logs.csv"
            }
        },
        "pubmed": {
            "func": fetch_pubmed_data,
            "kwargs": {
                "config_data": config_data,
                "INPUT_CSV": INPUT_CSV / "pubmed_search_schedule.csv",
                "OUTPUT_RECORDS": OUTPUT_DATASET_DIR / "pubmed_records.parquet",
                "OUTPUT_LOGS": OUTPUT_LOG_DIR / "pubmed_logs.csv"
            }
        },
        "curated": {
            "func": fetch_curated_data,
            "kwargs": {
                "INPUT_CSV": INPUT_CSV / "curated_search_schedule.csv",
                "OUTPUT_RECORDS": OUTPUT_DATASET_DIR / "curated_records.parquet",
                "OUTPUT_LOGS": OUTPUT_LOG_DIR / "curated_logs.csv"
            }
        }
    }

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_to_source = {
            executor.submit(task_info["func"], **task_info["kwargs"]): source_name
            for source_name, task_info in tasks.items()
        }

        for future in concurrent.futures.as_completed(future_to_source):
            source_name = future_to_source[future]

            try:
                future.result() 
                data_logger.info(f"Successfully completed fetching for: {source_name.upper()}")

            except Exception as exc:
                data_logger.error(f"{source_name.upper()} ingestion generated an exception: {exc}")
    
    total_time = time.perf_counter() - start_time
    data_logger.info(f"All parallel ingestion tasks finished in {total_time // 60} mins {(total_time / 60):.4f} s.")

if __name__ == "__main__":
    
    parallel_ingestion(
        config_data=config_data,
        INPUT_CSV=SEARCH_SCHEDULE_DIR,
        OUTPUT_DATASET_DIR=OUTPUT_DATASET_DIR,
        OUTPUT_LOG_DIR=OUTPUT_LOG_DIR
    )