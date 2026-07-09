import arxiv

import polars as pl

from pathlib import Path
from datetime import datetime

from utils.core_utils import (
    IST, get_logger,
    parse_nested_keywords, compute_query_plan, save_scraped_data
)

DELAY_SECONDS = 3.05

ARXIV_CLIENT = arxiv.Client(
    page_size=75,
    delay_seconds=DELAY_SECONDS,
    num_retries=3
)

arxiv_logger = get_logger("arxiv_fetcher")

def _build_arxiv_query(domain: str, keywords: list[str]) -> str:
    keyword_clauses = [f'all:"{kw.strip()}"' for kw in keywords if kw.strip()]
    
    if not keyword_clauses:
        return f"cat:{domain}"

    return f"cat:{domain} AND " + "(" + " OR ".join(keyword_clauses) + ")"

def _fetch_arxiv_papers(query_plan: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    records = []
    run_log = []

    for row in query_plan.iter_rows(named=True):
        domain = row["arxiv_category"]
        keywords = row["keyword_group"]
        cap = row["per_query_cap"]

        query_str = _build_arxiv_query(domain, keywords)
        
        search = arxiv.Search(
            query=query_str,
            max_results=cap,
            sort_by=arxiv.SortCriterion.Relevance
        )

        n_fetched = 0
        try:
            for result in ARXIV_CLIENT.results(search):
                records.append({
                    "id": result.entry_id.split("/")[-1],
                    "title": result.title.strip().replace("\n", " "),
                    "abstract": result.summary.strip().replace("\n", " "),
                    "authors": ", ".join(a.name for a in result.authors),
                    "doi": result.doi,
                    "url": result.pdf_url,
                    "source": "arXiv",
                    "source_domain": domain,
                    "source_keywords": ", ".join(keywords),
                    "source_query": query_str
                })
                n_fetched += 1

        except Exception as e:
            arxiv_logger.error(f"Query failed [{domain}] '{query_str[:60]}': {e}")

        run_log.append({
            "arxiv_category": domain,
            "keyword_group": ", ".join(keywords),
            "query": query_str,
            "requested_cap": cap,
            "results_fetched": n_fetched,
            "timestamp": datetime.now(IST)
        })

        arxiv_logger.info(f"[{domain}] fetched {n_fetched}/{cap} for keywords={keywords}")

    records = pl.DataFrame(records) if records else pl.DataFrame()
    run_log = pl.DataFrame(run_log)

    return records, run_log

def fetch_arxiv_data(
        config_data: pl.DataFrame,
        INPUT_CSV: Path, 
        OUTPUT_RECORDS: Path,
        OUTPUT_LOGS: Path
) -> None:
    
    arxiv_query_data = parse_nested_keywords(INPUT_CSV, "aggregated_keywords")
    arxiv_query_plan = compute_query_plan(
        config_data=config_data,
        query_data=arxiv_query_data,
        DOMAIN_COL="arxiv_category",
        logger=arxiv_logger
    )

    arxiv_logger.info(
        f"Query plan built: {arxiv_query_plan.height} throttled queries scheduled for arXiv. "
        f"Expected runtime >{arxiv_query_plan.height * 1.5 * DELAY_SECONDS / 60:.2f} minutes at {DELAY_SECONDS:.2f}s/query.\n"
    )

    arxiv_records, arxiv_logs = _fetch_arxiv_papers(arxiv_query_plan)

    save_scraped_data(
        arxiv_records, 
        arxiv_logs, 
        arxiv_logger,
        OUTPUT_RECORDS,
        OUTPUT_LOGS
    )