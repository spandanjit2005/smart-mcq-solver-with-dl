import os, re, json, time, logging

import polars as pl

from dotenv import load_dotenv

from Bio import Entrez, Medline

from pathlib import Path
from datetime import datetime, timezone

INPUT_CSV = "rag_pipeline/data/domains/pubmed_category.csv"
OUTPUT_PARQUET = Path("rag_pipeline/data/pubmed_corpus.parquet")
OUTPUT_LOG_CSV = Path("rag_pipeline/data/pubmed_log_run.csv")

SEED = 42

TOTAL_PAPERS_PER_CATEGORY = 150   # global budget per domain_name
MIN_RESULTS_PER_QUERY = 5         # floor, so no query returns nothing useful
MAX_QUERIES_PER_CATEGORY = 20     # cap on number of distinct keyword-groups queried per category
MAX_RESULTS_HARD_CAP = 150        # absolute ceiling per single API call

load_dotenv()
Entrez.email = os.getenv("ENTREZ_EMAIL")

ENTREZ_API_KEY = None

if ENTREZ_API_KEY:
    Entrez.api_key = ENTREZ_API_KEY

DELAY_SECONDS = 0.12 if ENTREZ_API_KEY else 0.35

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
    
    logger.info(f"Parsed nested keywords into single DataFrame.")
    return exploded_keywords

def select_diverse_keywords(keywords: list[str], k: int = 5) -> list[str]:
    if not keywords or len(keywords) <= k:
        return keywords
    
    def get_word_set(kw: str) -> set[str]:
        clean_text = re.sub(r'[\W_]+', ' ', kw.lower())
        return set(clean_text.split())
    
    keywords = sorted(keywords, key=len)
    selected = [keywords[0]]
    remaining = keywords[1:]
    
    while len(selected) < k and remaining:
        best_keyword = None
        min_overlap = float('inf')
        
        for kw in remaining:
            kw_words = get_word_set(kw)
            
            max_overlap_with_selected = max(
                len(kw_words.intersection(get_word_set(s))) 
                for s in selected
            )
            
            if max_overlap_with_selected < min_overlap:
                min_overlap = max_overlap_with_selected
                best_keyword = kw
                
        selected.append(best_keyword) # type: ignore
        remaining.remove(best_keyword) # type: ignore
        
    return selected

def build_pubmed_query(category: str, keywords: list[str]) -> str:
    keyword_clauses = [f'"{kw.strip()}"[tiab]' for kw in keywords if kw.strip()]
    
    if not keyword_clauses:
        return f"{category}"

    return f"{category} AND " + "(" + " OR ".join(keyword_clauses) + ")"

def compute_query_plan(query_data: pl.DataFrame) -> pl.DataFrame:
    plans = []

    for plan_config in query_data.partition_by("domain_name"):
        category = plan_config["domain_name"][0]
        n_groups = plan_config.height

        if n_groups > MAX_QUERIES_PER_CATEGORY:
            plan_config = plan_config.sample(n=MAX_QUERIES_PER_CATEGORY, seed=SEED)
            n_groups = MAX_QUERIES_PER_CATEGORY

        per_query_cap = max(MIN_RESULTS_PER_QUERY, TOTAL_PAPERS_PER_CATEGORY // n_groups)
        per_query_cap = min(per_query_cap, MAX_RESULTS_HARD_CAP)

        plan_config = plan_config.with_columns(pl.lit(per_query_cap).alias("per_query_cap"))
        plans.append(plan_config)

        logger.info(f"[{category}] Query plan computed: {n_groups} groups, cap={per_query_cap} papers/query.")

    logger.info(f"Successfully computed query plans for {len(plans)} categories.")
    
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
                        "source_query": query_str,
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
            "timestamp": datetime.now(timezone.utc),
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


def main() -> None:
    exploded_data = parse_nested_keywords(INPUT_CSV)
    logger.info(f"Exploded to {exploded_data.height} keyword-group query units across "
                f"{exploded_data["domain_name"].n_unique()} domains.")

    query_plan = compute_query_plan(exploded_data)
    logger.info(f"Query plan built: {query_plan.height} throttled queries scheduled "
                f"(~{query_plan.height * 1.5 * DELAY_SECONDS / 60:.2f} min runtime at {DELAY_SECONDS:.2f}s/query).")

    papers_data, log_data = fetch_papers(query_plan)
    save_scraped_data(papers_data, log_data)

if __name__ == "__main__":
    main()