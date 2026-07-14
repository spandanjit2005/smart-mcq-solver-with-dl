import re, logging

import polars as pl
from pathlib import Path

# Define function to parse nested keywords into a simple pl.DataFrame object
def parse_nested_keywords(csv_path: Path, KEYWORDS_COL: str) -> pl.DataFrame:
    df = pl.read_csv(csv_path)

    exploded_keywords = (
        df.with_columns(
            pl.col(KEYWORDS_COL)
            .str.json_decode(dtype=pl.List(pl.List(pl.String)))
            .alias("keyword_group")
        )
        .explode("keyword_group", empty_as_null=True)
        .filter(pl.col("keyword_group").is_not_null())
        .with_row_index("query_id")
    )

    return exploded_keywords

# Select top `k` semantically diverse keywords using Jaccard similarity
def _get_word_set(kw: str) -> set[str]:
    """Cleans and tokenizes a keyword into a set of words."""
    clean_text = re.sub(r'[\W_]+', ' ', kw.lower())

    return set(clean_text.split())

def _jaccard_similarity(set1: set[str], set2: set[str]) -> float:
    """Calculates true Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)

def select_diverse_keywords(keywords: list[str], k: int = 5) -> list[str]:
    """Select top `k` semantically diverse keywords using Jaccard similarity."""
    if not keywords or len(keywords) <= k:
        return keywords

    keywords = sorted(keywords, key=len, reverse=True)
    selected = [keywords[0]]
    remaining = keywords[1:]

    word_sets = {kw: _get_word_set(kw) for kw in keywords}

    while len(selected) < k and remaining:
        best_keyword = remaining[0]
        min_max_similarity = float('inf')

        for kw in remaining:
            kw_words = word_sets[kw]
            max_sim_with_selected = max(
                _jaccard_similarity(kw_words, word_sets[s]) for s in selected
            )

            if max_sim_with_selected < min_max_similarity:
                min_max_similarity = max_sim_with_selected
                best_keyword = kw

        selected.append(best_keyword)
        remaining.remove(best_keyword)

    return selected

def select_diverse_keywords_clustered(clusters: list[list[str]], k: int = 5) -> list[str]:
    flat = [kw for cluster in clusters for kw in cluster]
    
    return select_diverse_keywords(flat, k)

# Define generic function to compute arXiv and PubMed query plans
def compute_query_plan(
        config_data: dict[str, int], 
        query_data: pl.DataFrame, 
        DOMAIN_COL: str,
        logger: logging.Logger
) -> pl.DataFrame:
    
    TOTAL_RESULTS_PER_DOMAIN = config_data["TOTAL_RESULTS_PER_DOMAIN"]
    MAX_QUERIES_PER_DOMAIN = config_data["MAX_QUERIES_PER_DOMAIN"]
    MIN_RESULTS_PER_QUERY = config_data["MIN_RESULTS_PER_QUERY"]
    MAX_RESULTS_HARD_CAP = config_data["MAX_RESULTS_HARD_CAP"]

    plans = []

    for plan_config in query_data.partition_by(DOMAIN_COL):
        domain = plan_config[DOMAIN_COL][0]
        
        plan_config = plan_config.with_columns(
            pl.col("keyword_group").map_elements(
                lambda kw_series: select_diverse_keywords(kw_series.to_list(), k=5),
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

        per_query_cap = max(MIN_RESULTS_PER_QUERY, TOTAL_RESULTS_PER_DOMAIN // n_groups)
        per_query_cap = min(per_query_cap, MAX_RESULTS_HARD_CAP)

        plan_config = plan_config.with_columns(pl.lit(per_query_cap).alias("per_query_cap"))
        plans.append(plan_config)

        logger.info(f"[{domain}] Query plan computed: {n_groups} diverse groups, cap={per_query_cap} papers/query.")

    logger.info(f"Successfully computed query plans for {len(plans)} domains for arXiv." if DOMAIN_COL == "arxiv_category" else
                f"Successfully computed query plans for {len(plans)} domains for PubMed.")
    
    return pl.concat(plans)

# Define function to save scraped data
def save_scraped_data(
        records: pl.DataFrame, 
        log_run: pl.DataFrame, 
        logger: logging.Logger,
        OUTPUT_PARQUET: Path, 
        OUTPUT_LOG: Path
) -> None:
    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_LOG.parent.mkdir(parents=True, exist_ok=True)

    log_run.write_csv(OUTPUT_LOG)

    if records.height == 0:
        logger.warning("No records collected.")
        return

    deduped = (
        records.unique(subset=["id"], keep="first").sort("source_domain")
    )

    deduped.write_parquet(OUTPUT_PARQUET, compression="zstd")

    logger.info(f"Saved {deduped.height} unique records -> {OUTPUT_PARQUET}")