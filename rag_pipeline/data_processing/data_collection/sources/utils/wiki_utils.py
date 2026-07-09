import os, time, requests, wikipediaapi

import polars as pl

from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

from utils.core_utils import (
    IST, get_logger,
    parse_nested_keywords, select_diverse_keywords_clustered, save_scraped_data
)

DELAY_SECONDS = 1.0

load_dotenv()
USER_AGENT = f"RAGCorpusHarvester/1.0 ({os.getenv("API_CONTACT_EMAIL")})"
WIKI_CLIENT = wikipediaapi.Wikipedia(
    user_agent=USER_AGENT,
    language='en',
    extract_format=wikipediaapi.ExtractFormat.WIKI
)

wiki_logger = get_logger("wiki_fetcher")

def _get_exact_wiki_title(query: str) -> str | None:
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
        wiki_logger.error(f"Search API error for '{query}': {e}")
        
    return None

def _load_curated_titles(curated_path: Path) -> set[str]:
    df = pl.read_csv(curated_path)
    titles = set()

    for u in df["url"]:
        if u.startswith("http"):
            continue  

        page = u.split("#")[0].replace("_", " ").strip()
        titles.add(page.casefold())

    return titles

def _compute_wiki_query_plan(config_data: pl.DataFrame, query_data: pl.DataFrame) -> pl.DataFrame:
    MAX_QUERIES_PER_DOMAIN = config_data["MAX_QUERIES_PER_DOMAIN"].item()
    
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

    wiki_logger.info(f"Successfully computed query plans for {len(plans)} domains for Wikipedia.")

    return pl.DataFrame(plans)

def _fetch_wiki_pages(
        query_plan: pl.DataFrame,
        seen_titles: set[str] | None
) -> tuple[pl.DataFrame, pl.DataFrame]:
    
    records = []
    run_log = []
    seen_titles = set(seen_titles) if seen_titles else set()

    for row in query_plan.iter_rows(named=True):
        domain = row["domain_name"]
        queries = row["search_queries"]

        n_fetched = 0
        wiki_logger.info(f"[{domain}] Resolving and fetching up to {len(queries)} Wikipedia pages.")

        for query in queries:
            try:
                exact_title = _get_exact_wiki_title(query)
                time.sleep(DELAY_SECONDS)

                if not exact_title:
                    continue

                norm_title = exact_title.casefold()
                if norm_title in seen_titles:
                    continue

                page = WIKI_CLIENT.page(exact_title)

                if page.exists():
                    records.append({
                        "id": page.pageid,
                        "title": page.title,
                        "abstract": page.summary.strip().replace("\n", " "),
                        "full_text": page.text.strip(),
                        "url": page.fullurl,
                        "source": "Wikipedia",
                        "source_domain": domain,
                        "source_query": query
                    })

                    seen_titles.add(norm_title)
                    n_fetched += 1
                    wiki_logger.info(f"Successfully fetched: {page.title}")

            except Exception as e:
                wiki_logger.error(f"Failed to fetch '{query}' for domain [{domain}]: {e}")

        run_log.append({
            "domain_name": domain,
            "queries_attempted": ", ".join(queries),
            "results_fetched": n_fetched,
            "timestamp": datetime.now(IST)
        })
        wiki_logger.info(f"[{domain}] fetched {n_fetched} for keywords={queries}")

    records = pl.DataFrame(records) if records else pl.DataFrame()
    run_log = pl.DataFrame(run_log)
    return records, run_log

def fetch_wiki_data(
        config_data: pl.DataFrame,
        INPUT_CSV: Path, 
        CURATED_CSV: Path,
        OUTPUT_RECORDS: Path,
        OUTPUT_LOGS: Path
) -> None:
    
    wiki_query_data = parse_nested_keywords(INPUT_CSV, "domain_keywords")
    wiki_query_plan = _compute_wiki_query_plan(
        config_data=config_data, 
        query_data=wiki_query_data
    )
    curated_titles = _load_curated_titles(CURATED_CSV)

    wiki_logger.info(
        f"Query plan built: {wiki_query_plan.height} throttled queries scheduled for Wikipedia. "
        f"Expected runtime >{wiki_query_plan.height * 10.0 * DELAY_SECONDS / 60:.2f} minutes at {DELAY_SECONDS:.2f}s/query.\n"
    )

    wiki_records, wiki_logs = _fetch_wiki_pages(wiki_query_plan, curated_titles)

    save_scraped_data(
        wiki_records, 
        wiki_logs, 
        wiki_logger,
        OUTPUT_RECORDS,
        OUTPUT_LOGS
    )