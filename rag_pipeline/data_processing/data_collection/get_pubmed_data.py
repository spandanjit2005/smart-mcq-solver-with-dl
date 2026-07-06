import os, re, json, time, logging

import polars as pl

from dotenv import load_dotenv

from Bio import Entrez, Medline

from pathlib import Path
from datetime import datetime, timezone, timedelta

INPUT_CSV = "rag_pipeline/data/domains/pubmed_search_schedule.csv"
OUTPUT_PARQUET = Path("rag_pipeline/data/pubmed_corpus.parquet")
OUTPUT_LOG_CSV = Path("rag_pipeline/data/pubmed_log_run.csv")

SEED = 42

TOTAL_PAPERS_PER_DOMAIN = 150   # global budget per domain_name
MAX_QUERIES_PER_DOMAIN = 30     # cap on number of distinct keyword-groups queried per category
MIN_RESULTS_PER_QUERY = 5       # floor, so no query returns nothing useful
MAX_RESULTS_HARD_CAP = 150      # absolute ceiling per single API call

load_dotenv()
Entrez.email = os.getenv("API_CONTACT_EMAIL")

ENTREZ_API_KEY = None

if ENTREZ_API_KEY:
    Entrez.api_key = ENTREZ_API_KEY

DELAY_SECONDS = 0.12 if ENTREZ_API_KEY else 0.35

IST = timezone(timedelta(hours=5, minutes=30))

logging.Formatter.converter = staticmethod(
    lambda ts: datetime.fromtimestamp(ts, tz=IST).timetuple()
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("pubmed_harvester")

def parse_nested_keywords(csv_path: str) -> pl.DataFrame:
    df = pl.read_csv(csv_path)

    exploded_keywords = (
        df.with_columns(
            pl.col("domain_keywords")
            .str.json_decode(dtype=pl.List(pl.List(pl.Utf8)))
            .alias("keyword_group")
        )
        .explode("keyword_group", empty_as_null=True)
        .filter(pl.col("keyword_group").is_not_null())
        .with_row_index("query_id")
    )
    
    return exploded_keywords

def get_word_set(kw: str) -> set[str]:
    clean_text = re.sub(r'[\W_]+', ' ', kw.lower())

    return set(clean_text.split())

def select_diverse_keywords(keywords: list[str], k: int = 5) -> list[str]:
    keywords = list(keywords)
    
    if not keywords or len(keywords) <= k:
        return keywords

    keywords = sorted(keywords, key=len, reverse=True)
    selected = [keywords[0]]
    remaining = keywords[1:]

    word_sets = {kw: get_word_set(kw) for kw in keywords}

    while len(selected) < k and remaining:
        best_keyword = None
        min_overlap = float('inf')

        for kw in remaining:
            kw_words = word_sets[kw]
            max_overlap_with_selected = max(
                len(kw_words & word_sets[s]) for s in selected
            )
            if max_overlap_with_selected < min_overlap:
                min_overlap = max_overlap_with_selected
                best_keyword = kw

        selected.append(best_keyword) # type: ignore 
        remaining.remove(best_keyword) # type: ignore

    return selected

def build_pubmed_query(domain: str, keywords: list[str]) -> str:
    keyword_clauses = [f'"{kw.strip()}"[tiab]' for kw in keywords if kw.strip()]
    
    if not keyword_clauses:
        return f"{domain}"

    return f"{domain} AND " + "(" + " OR ".join(keyword_clauses) + ")"

def compute_query_plan(query_plan: pl.DataFrame) -> pl.DataFrame:
    plans = []

    for plan_config in query_plan.partition_by("domain_name"):
        domain = plan_config["domain_name"][0]
        
        plan_config = plan_config.with_columns(
            pl.col("keyword_group").map_elements(
                lambda kw_list: select_diverse_keywords(kw_list, k=5),
                return_dtype=pl.List(pl.String)
            ).alias("keyword_group")
        )

        n_groups = plan_config.height

        if n_groups > MAX_QUERIES_PER_DOMAIN:
            temp_df = plan_config.with_columns(
                pl.col("keyword_group").list.join(" ").alias("__kw_str")
            )

            kw_strings = temp_df["__kw_str"].to_list()
            diverse_row_strings = select_diverse_keywords(kw_strings, k=MAX_QUERIES_PER_DOMAIN)
            plan_config = temp_df.filter(pl.col("__kw_str").is_in(diverse_row_strings)).drop("__kw_str")
            
            if plan_config.height > MAX_QUERIES_PER_DOMAIN:
                plan_config = plan_config.head(MAX_QUERIES_PER_DOMAIN)
                
            n_groups = plan_config.height

        if n_groups == 0:
            logger.warning(f"[{domain}] No query groups survived selection; skipping domain.")
            continue

        per_query_cap = max(MIN_RESULTS_PER_QUERY, TOTAL_PAPERS_PER_DOMAIN // n_groups)
        per_query_cap = min(per_query_cap, MAX_RESULTS_HARD_CAP)

        plan_config = plan_config.with_columns(pl.lit(per_query_cap).alias("per_query_cap"))
        plans.append(plan_config)

        logger.info(f"[{domain}] Query plan computed: {n_groups} diverse groups, cap={per_query_cap} papers/query.")

    logger.info(f"Successfully computed query plans for {len(plans)} domains.")
    
    return pl.concat(plans)

def fetch_papers(query_plan: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    records = []
    run_log = []

    for row in query_plan.iter_rows(named=True):
        domain = row["domain_name"]
        keywords = row["keyword_group"]
        cap = row["per_query_cap"]

        query_str = build_pubmed_query(domain, keywords)

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
                        "record_id": pmid,
                        "title": rec.get("TI", "").strip(),
                        "abstract": rec.get("AB", "").strip(),
                        "authors": ", ".join(rec.get("AU", [])),
                        "journal": rec.get("JT", ""),
                        "pub_date": rec.get("DP", ""),
                        "doi": next(
                            (v.split(" ")[0] for v in rec.get("AID", []) if "[doi]" in v),
                            None,
                        ),
                        "pmcid": rec.get("PMC", None),
                        "mesh_terms": ", ".join(rec.get("MH", [])),
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
                        "source_domain": domain,
                        "source_keywords": ", ".join(keywords),
                        "source_query": query_str
                    })
                    
                n_fetched = len(medline_records)

        except Exception as e:
            logger.error(f"Query failed [{domain}] '{query_str[:60]}': {e}")

        run_log.append({
            "domain_name": domain,
            "keyword_group": ", ".join(keywords),
            "query": query_str,
            "requested_cap": cap,
            "n_fetched": n_fetched,
            "timestamp": datetime.now(IST)
        })

        logger.info(f"[{domain}] fetched {n_fetched}/{cap} for keywords={keywords}")

    records = pl.DataFrame(records) if records else pl.DataFrame()
    run_log = pl.DataFrame(run_log)

    return records, run_log

def save_scraped_data(papers: pl.DataFrame, log_run: pl.DataFrame) -> None:
    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)

    if papers.height == 0:
        logger.warning("No papers collected.")
        return

    deduped = (
        papers.filter(pl.col("record_id").is_not_null() & (pl.col("record_id") != ""))
        .unique(subset=["record_id"], keep="first")
        .sort(["source_domain"])
    )

    deduped.write_parquet(OUTPUT_PARQUET, compression="zstd")
    log_run.write_csv(OUTPUT_LOG_CSV)

    logger.info(f"Saved {deduped.height} unique papers -> {OUTPUT_PARQUET}")
    logger.info(f"Category distribution:\n{deduped.group_by("source_domain").len().sort('len', descending=True)}")

def main():
    query_data = parse_nested_keywords(INPUT_CSV)

    query_plan = compute_query_plan(query_data)
    logger.info(f"Query plan built: {query_plan.height} throttled queries scheduled "
                f"(>{query_plan.height * 10.0 * DELAY_SECONDS / 60:.2f} min runtime at {DELAY_SECONDS:.2f}s/query).")

    papers_data, log_data = fetch_papers(query_plan)
    save_scraped_data(papers_data, log_data)

if __name__ == "__main__":
    main()