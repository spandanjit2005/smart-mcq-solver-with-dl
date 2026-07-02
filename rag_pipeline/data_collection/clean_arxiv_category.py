import polars as pl

arxiv_data = pl.read_csv("rag_pipeline/data/domains/arxiv_category.csv")  
count_data = pl.read_csv("rag_pipeline/data/domains/domain_count_keyword.csv")

arxiv_keyword_data = arxiv_data.join(
    count_data.select(["domain_name", "domain_keywords"]),
    on="domain_name",
    how="left"
)

arxiv_keyword_cleaned = (
    arxiv_keyword_data
    .with_columns(pl.col("arxiv_category").str.split(";"))
    .explode("arxiv_category", empty_as_null=True)
    .with_columns(pl.col("arxiv_category").str.strip_chars())
    
    .with_columns(pl.col("domain_keywords").str.split(";"))
    .explode("domain_keywords", empty_as_null=True)
    .with_columns(pl.col("domain_keywords").str.strip_chars())
    
    .filter(
        pl.col("arxiv_category").is_not_null() & (pl.col("arxiv_category") != "") &
        pl.col("domain_keywords").is_not_null() & (pl.col("domain_keywords") != "")
    )
)

category_search_schedule = arxiv_keyword_cleaned.group_by("arxiv_category").agg(
    pl.col("domain_keywords").unique().implode().list.join(", ").alias("aggregated_keywords"),
    pl.col("domain_name").unique().implode().list.join("; ").alias("contributing_domains")
)

category_search_schedule.write_csv("rag_pipeline/data/domains/arxiv_search_schedule.csv")

print(category_search_schedule.sample(10))
