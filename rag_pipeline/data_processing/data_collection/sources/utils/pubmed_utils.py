import os

import polars as pl

from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

from Bio import Entrez, Medline

from utils.core_utils import (
    IST, get_logger,
    parse_nested_keywords, compute_query_plan, save_scraped_data
)

load_dotenv()
Entrez.email = os.getenv("API_CONTACT_EMAIL")

ENTREZ_API_KEY = None

if ENTREZ_API_KEY:
    Entrez.api_key = ENTREZ_API_KEY

DELAY_SECONDS = 0.12 if ENTREZ_API_KEY else 0.35

pubmed_logger = get_logger("pubmed_fetcher")

def _build_pubmed_query(domain: str, keywords: list[str]) -> str:
    keyword_clauses = [f'"{kw.strip()}"[tiab]' for kw in keywords if kw.strip()]
    
    if not keyword_clauses:
        return f"{domain}"

    return f"{domain} AND " + "(" + " OR ".join(keyword_clauses) + ")"

def _fetch_pubmed_papers(query_plan: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    records = []
    run_log = []

    for row in query_plan.iter_rows(named=True):
        domain = row["domain_name"]
        keywords = row["keyword_group"]
        cap = row["per_query_cap"]

        query_str = _build_pubmed_query(domain, keywords)

        n_fetched = 0

        try:
            handle = Entrez.esearch(
                db="pubmed",
                term=query_str,
                retmax=cap,
                sort="relevance"
            )
            search_result = Entrez.read(handle)
            handle.close()

            id_list = search_result.get("IdList", []) # type: ignore

            if id_list:
                handle = Entrez.efetch(
                    db="pubmed",
                    id=id_list,
                    rettype="medline",
                    retmode="text"
                )
                medline_records = list(Medline.parse(handle))
                handle.close()

                for rec in medline_records:
                    pmid = rec.get("PMID", "")
                    records.append({
                        "id": pmid,
                        "title": rec.get("TI", "").strip(),
                        "abstract": rec.get("AB", "").strip(),
                        "authors": ", ".join(rec.get("AU", [])),
                        "doi": next(
                            (v.split(" ")[0] for v in rec.get("AID", []) if "[doi]" in v),
                            None,
                        ),
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
                        "source": "PubMed",
                        "source_domain": domain,
                        "source_keywords": ", ".join(keywords),
                        "source_query": query_str
                    })
                    
                n_fetched = len(medline_records)

        except Exception as e:
            pubmed_logger.error(f"Query failed [{domain}] '{query_str[:60]}': {e}")

        run_log.append({
            "domain_name": domain,
            "keyword_group": ", ".join(keywords),
            "query": query_str,
            "requested_cap": cap,
            "results_fetched": n_fetched,
            "timestamp": datetime.now(IST)
        })

        pubmed_logger.info(f"[{domain}] fetched {n_fetched}/{cap} for keywords={keywords}")

    records = pl.DataFrame(records) if records else pl.DataFrame()
    run_log = pl.DataFrame(run_log)

    return records, run_log

def fetch_pubmed_data(
        config_data: pl.DataFrame,
        INPUT_CSV: Path, 
        OUTPUT_RECORDS: Path,
        OUTPUT_LOGS: Path
) -> None:
    
    pubmed_query_data = parse_nested_keywords(INPUT_CSV, "domain_keywords")
    pubmed_query_plan = compute_query_plan(
        config_data=config_data,
        query_data=pubmed_query_data,
        DOMAIN_COL="domain_name",
        logger=pubmed_logger
    )

    pubmed_logger.info(
        f"Query plan built: {pubmed_query_plan.height} throttled queries scheduled for PubMed. "
        f"Expected runtime >{pubmed_query_plan.height * 10.0 * DELAY_SECONDS / 60:.2f} minutes at {DELAY_SECONDS:.2f}s/query.\n"
    )

    pubmed_records, pubmed_logs = _fetch_pubmed_papers(pubmed_query_plan)

    save_scraped_data(
        pubmed_records, 
        pubmed_logs, 
        pubmed_logger,
        OUTPUT_RECORDS,
        OUTPUT_LOGS
    )