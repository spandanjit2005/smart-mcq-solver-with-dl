import os, re, json, time, requests, wikipediaapi, logging

import polars as pl

from dotenv import load_dotenv

from pathlib import Path
from datetime import datetime, timezone

INPUT_CSV = "rag_pipeline/data/domains/domain_count_keyword.csv"
OUTPUT_PARQUET = Path("rag_pipeline/data/wikipedia_corpus.parquet")
OUTPUT_LOG_CSV = Path("rag_pipeline/data/wikipedia_log_run.csv")

SEED = 42

DELAY_SECONDS = 1.0

MAX_QUERIES_PER_DOMAIN = 5

load_dotenv()

USER_AGENT = f"RAGCorpusHarvester/1.0 ({os.getenv("API_CONTACT_EMAIL")})"
CLIENT = wikipediaapi.Wikipedia(
    user_agent=USER_AGENT,
    language='en',
    extract_format=wikipediaapi.ExtractFormat.WIKI
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("arxiv_harvester")

def parse_nested_keywords(csv_path: str) -> pl.DataFrame:
    df = pl.read_csv(csv_path)

    exploded_keywords = (
        df.with_columns(
            pl.col("domain_keywords")
            .str.json_decode(dtype=pl.List(pl.List(pl.String)))
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

def select_diverse_keywords_clustered(clusters: list[list[str]], k: int = 5) -> list[str]:
    flat = [kw for cluster in clusters for kw in cluster]
    
    return select_diverse_keywords(flat, k)

def get_exact_wikipedia_title(query: str) -> str | None:
    search_url = "https://en.wikipedia.org/w/api.php"

    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "utf8": "1",
        "srlimit": 1
    }

    try:
        response = requests.get(search_url, params=params, headers={"User-Agent": USER_AGENT})
        response.raise_for_status()
        data = response.json()
        search_results = data.get("query", {}).get("search", [])

        if search_results:
            return search_results[0]["title"]
        
    except Exception as e:
        logger.error(f"Search API error for '{query}': {e}")
        
    return None

def compute_query_plan(query_data: pl.DataFrame) -> pl.DataFrame:
    plans = []

    grouped = query_data.group_by("domain_name", maintain_order=True).agg(
        pl.col("keyword_group").alias("keyword_clusters")
    )

    for row in grouped.iter_rows(named=True):
        domain_name = row["domain_name"]
        keyword_clusters = row["keyword_clusters"]

        if not keyword_clusters:
            keyword_clusters = []

        diverse_keywords = select_diverse_keywords_clustered(keyword_clusters, k=MAX_QUERIES_PER_DOMAIN - 1)

        search_queries = [domain_name] + diverse_keywords

        plans.append({
            "domain_name": domain_name,
            "search_queries": search_queries
        })

    logger.info(f"Successfully computed query plans for {len(plans)} domains.")

    return pl.DataFrame(plans)

def fetch_pages(query_plan: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    records = []
    run_log = []

    for row in query_plan.iter_rows(named=True):
        domain = row["domain_name"]
        queries = row["search_queries"]

        n_fetched = 0
        fetched_titles = set()

        logger.info(f"[{domain}] Resolving and fetching up to {len(queries)} Wikipedia pages.")

        for query in queries:
            
            try:
                exact_title = get_exact_wikipedia_title(query)
                time.sleep(DELAY_SECONDS) 
                
                if not exact_title or exact_title in fetched_titles:
                    continue
                    
                page = CLIENT.page(exact_title)
                
                if page.exists():
                    records.append({
                        "page_id": page.pageid,
                        "title": page.title,
                        "summary": page.summary.strip().replace("\n", " "),
                        "full_text": page.text.strip(),
                        "url": page.fullurl,
                        "source_domain": domain,
                        "source_query": query
                    })

                    fetched_titles.add(page.title)
                    n_fetched += 1

                    logger.info(f"Successfully Fetched: {page.title}")
                
            except Exception as e:
                logger.error(f"Failed to fetch '{query}' for domain [{domain}]: {e}")
                
        run_log.append({
            "domain_name": domain,
            "queries_attempted": ", ".join(queries),
            "pages_fetched": n_fetched,
            "timestamp": datetime.now(timezone.utc),
        })

        logger.info(f"[{domain}] fetched {n_fetched} for keywords={queries}")

    records = pl.DataFrame(records) if records else pl.DataFrame()
    run_log = pl.DataFrame(run_log)

    return records, run_log

def save_scraped_data(pages: pl.DataFrame, log_run: pl.DataFrame) -> None:
    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)

    if pages.height == 0:
        logger.warning("No pages collected.")
        return

    deduped = (
        pages.unique(subset=["page_id"], keep="first").sort("source_domain")
    )

    deduped.write_parquet(OUTPUT_PARQUET, compression="zstd")
    log_run.write_csv(OUTPUT_LOG_CSV)

    logger.info(f"Saved {deduped.height} unique pages -> {OUTPUT_PARQUET}")
    logger.info(f"Category distribution:\n{deduped.group_by("source_domain").len().sort('len', descending=True)}")

def main():
    query_data = parse_nested_keywords(INPUT_CSV)

    query_plan = compute_query_plan(query_data)
    logger.info(f"Query plan built: {query_plan.height} throttled queries scheduled "
                f"(>{query_plan.height * 14.0 * DELAY_SECONDS / 60:.2f} min runtime at {DELAY_SECONDS:.2f}s/query).")

    page_data, log_data = fetch_pages(query_plan)
    save_scraped_data(page_data, log_data)

if __name__ == "__main__":
    main()