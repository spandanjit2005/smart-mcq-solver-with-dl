import re, json, time, arxiv, logging

import polars as pl

from pathlib import Path
from datetime import datetime, timezone

INPUT_CSV = "rag_pipeline/data/domains/arxiv_search_schedule.csv"
OUTPUT_PARQUET = Path("rag_pipeline/data/arxiv_corpus.parquet")
OUTPUT_LOG_CSV = Path("rag_pipeline/data/arxiv_log_run.csv")

SEED = 42

DELAY_SECONDS = 3.05

TOTAL_PAPERS_PER_CATEGORY = 150   # global budget per arxiv_category
MIN_RESULTS_PER_QUERY = 5         # floor, so no query returns nothing useful
MAX_QUERIES_PER_CATEGORY = 20     # cap on number of distinct keyword-groups queried per category
MAX_RESULTS_HARD_CAP = 100        # absolute ceiling per single API call

CLIENT = arxiv.Client(
    page_size=75,
    delay_seconds=DELAY_SECONDS,
    num_retries=5
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("arxiv_harvester")

def parse_nested_keywords(csv_path: str) -> pl.DataFrame:
    df = pl.read_csv(csv_path)

    exploded_keywords = (
        df.with_columns(
            pl.col("aggregated_keywords")
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

def build_arxiv_query(category: str, keywords: list[str]) -> str:
    keyword_clauses = [f'all:"{kw.strip()}"' for kw in keywords if kw.strip()]
    
    if not keyword_clauses:
        return f"cat:{category}"

    return f"cat:{category} AND " + "(" + " OR ".join(keyword_clauses) + ")"

def compute_query_plan(keywords: pl.DataFrame) -> pl.DataFrame:
    plans = []

    for plan_config in keywords.partition_by("arxiv_category"):
        category = plan_config["arxiv_category"][0]
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
        category = row["arxiv_category"]
        keywords = row["keyword_group"]
        cap = row["per_query_cap"]

        query_str = build_arxiv_query(category, keywords)
        search = arxiv.Search(
            query=query_str,
            max_results=cap,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        n_fetched = 0
        try:
            for result in CLIENT.results(search):
                records.append({
                    "entry_id": result.entry_id,
                    "title": result.title.strip().replace("\n", " "),
                    "summary": result.summary.strip().replace("\n", " "),
                    "authors": ", ".join(a.name for a in result.authors),
                    "primary_category": result.primary_category,
                    "categories": ", ".join(result.categories),
                    "published": result.published,
                    "updated": result.updated,
                    "pdf_url": result.pdf_url,
                    "doi": result.doi,
                    "source_category": category,
                    "source_keywords": ", ".join(keywords),
                    "source_query": query_str,
                })
                n_fetched += 1

        except Exception as e:
            logger.error(f"Query failed [{category}] '{query_str[:60]}': {e}")

        run_log.append({
            "arxiv_category": category,
            "keyword_group": ", ".join(keywords),
            "query": query_str,
            "requested_cap": cap,
            "n_fetched": n_fetched,
            "timestamp": datetime.now(timezone.utc),
        })

        logger.info(f"[{category}] fetched {n_fetched}/{cap} for keywords={keywords}")

    records = pl.DataFrame(records) if records else pl.DataFrame()
    run_log = pl.DataFrame(run_log)

    return records, run_log

def save_scraped_data(papers: pl.DataFrame, log_run: pl.DataFrame) -> None:
    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)

    if papers.height == 0:
        logger.warning("No papers collected.")
        return

    deduped = (
        papers.unique(subset=["entry_id"], keep="first")
        .with_columns([
            pl.col("published").cast(pl.Datetime),
            pl.col("updated").cast(pl.Datetime),
        ])
        .sort(["source_category", "published"], descending=[False, True])
    )

    deduped.write_parquet(OUTPUT_PARQUET, compression="zstd")
    log_run.write_csv(OUTPUT_LOG_CSV)

    logger.info(f"Saved {deduped.height} unique papers -> {OUTPUT_PARQUET}")
    logger.info(f"Category distribution:\n{deduped.group_by("source_category").len().sort('len', descending=True)}")

def main() -> None:
    exploded_data = parse_nested_keywords(INPUT_CSV)
    logger.info(f"Exploded to {exploded_data.height} keyword-group query units across "
                f"{exploded_data["arxiv_category"].n_unique()} categories.")

    query_plan = compute_query_plan(exploded_data)
    logger.info(f"Query plan built: {query_plan.height} throttled queries scheduled "
                f"(~{query_plan.height * 1.5 * DELAY_SECONDS / 60:.2f} min runtime at {DELAY_SECONDS:.2f}s/query).")

    papers_data, log_data = fetch_papers(query_plan)
    save_scraped_data(papers_data, log_data)

if __name__ == "__main__":
    main()