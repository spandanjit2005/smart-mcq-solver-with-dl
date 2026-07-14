import os, sys, time, signal
import concurrent.futures

from pathlib import Path

project_root = str(Path(__file__).resolve().parents[4])

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from rag_pipeline.utils.config import get_logger, data_paths, data_ingestion_config

from utils.arxiv_utils import fetch_arxiv_data
from utils.pubmed_utils import fetch_pubmed_data
from utils.wiki_utils import fetch_wiki_data
from utils.curated_utils import fetch_curated_data

SEARCH_SCHEDULE_DIR = data_paths["search_schedule_path"]
OUTPUT_DATASET_DIR = data_paths["dataset_path"]
OUTPUT_LOG_DIR = data_paths["collection_log_path"]

data_logger = get_logger("all_data")

def force_kill_handler(signum, frame):
    data_logger.warning("KeyboardInterrupt. Forcefully shutting down parallel data ingestion.")
    os._exit(1)

def parallel_ingestion(
    config_data: dict[str, str],
    INPUT_CSV: Path,
    OUTPUT_DATASET_DIR: Path,
    OUTPUT_LOG_DIR: Path,
) -> None:
    
    OUTPUT_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_LOG_DIR.mkdir(parents=True, exist_ok=True)

    signal.signal(signal.SIGINT, force_kill_handler)

    data_logger.info("Starting parallel data ingestion.")
    start_time = time.perf_counter()

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

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
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
    data_logger.info(f"All parallel ingestion tasks finished in {int(total_time // 60)} mins {(total_time / 60):.4f} s.")

if __name__ == "__main__":
    
    parallel_ingestion(
        config_data=data_ingestion_config,
        INPUT_CSV=SEARCH_SCHEDULE_DIR,
        OUTPUT_DATASET_DIR=OUTPUT_DATASET_DIR,
        OUTPUT_LOG_DIR=OUTPUT_LOG_DIR
    )