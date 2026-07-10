import os, re, time, requests, arxiv, wikipediaapi, logging

import polars as pl

from pathlib import Path
from datetime import datetime

import urllib.parse
from collections import defaultdict

from bs4 import BeautifulSoup

from rag_pipeline.utils.config import IST, get_logger
from utils.core_utils import save_scraped_data

from utils.arxiv_utils import ARXIV_CLIENT
from utils.wiki_utils import DELAY_SECONDS, USER_AGENT, WIKI_CLIENT


ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5}(v\d+)?)")

REQUEST_HEADERS = {"User-Agent": USER_AGENT}

curated_logger = get_logger("curated_fetcher")

def _load_curated_schedule(path: Path) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    wiki_targets = defaultdict(list)
    external_urls = defaultdict(list)

    data = pl.read_csv(path)

    for (raw,) in data.select("url").iter_rows():
        if raw.startswith("http"):

            if "arxiv.org" in raw:
                external_urls["arxiv"].append(raw)
            elif "plato.stanford.edu" in raw:
                external_urls["plato_stanford"].append(raw)
            elif "nobelprize.org" in raw:
                external_urls["nobelprize"].append(raw)
            else:
                external_urls["unknown"].append(raw)

            continue

        decoded = urllib.parse.unquote(raw)

        if "#" in decoded:
            title, anchor = decoded.split("#", 1)
        else:
            title, anchor = decoded, None

        title = title.replace("_", " ").strip()

        if anchor:
            wiki_targets[title].append(anchor.replace("_", " "))
        else:
            wiki_targets.setdefault(title, [])

    return wiki_targets, external_urls


def _fetch_curated_wiki(wiki_targets: dict[str, list[str]]) -> tuple[list[dict], int]:
    
    records = []
    n_fetched = 0

    import itertools
    for title, _ in wiki_targets.items():
        try:

            page = WIKI_CLIENT.page(title)
            time.sleep(DELAY_SECONDS)

            if not page.exists():
                curated_logger.warning(f"Curated Wikipedia title not found: {title}")
                continue

            records.append({
                "id": str(page.pageid),
                "title": page.title,
                "abstract": page.summary.strip().replace("\n", " "),
                "full_text": page.text.strip(),
                "url": page.fullurl,
                "source": "Curated Wikipedia",
                "source_domain": title,
                "source_query": title
            })

            n_fetched += 1
            curated_logger.info(f"Successfully fetched curated Wikipedia page: {page.title}")

        except Exception as e:
            curated_logger.error(f"Failed to fetch curated Wikipedia title '{title}': {e}")

    return records, n_fetched


def _fetch_curated_arxiv(urls: list[str]) -> tuple[list[dict], int]:
    
    records = []
    n_fetched = 0

    for url in urls:

        match = ARXIV_ID_RE.search(url)

        if not match:
            curated_logger.error(f"Could not parse arxiv ID from URL: {url}")
            continue

        arxiv_id = match.group(1)

        try:
            search = arxiv.Search(id_list=[arxiv_id])
            result = next(ARXIV_CLIENT.results(search))

            records.append({
                "id": result.entry_id.split("/")[-1],
                "title": result.title.strip(),
                "abstract": result.summary.strip().replace("\n", " "),
                "authors": ", ".join(a.name for a in result.authors),
                "doi": result.doi,
                "url": result.pdf_url,
                "source": "Curated arXiv",
                "source_domain": "General Relativity",
                "source_query": result.title.strip()
            })
            
            n_fetched += 1
            curated_logger.info(f"Successfully fetched curated arxiv paper: {result.title.strip()}")

        except Exception as e:
            curated_logger.error(f"Failed to fetch curated arxiv URL '{url}': {e}")

    return records, n_fetched


def _clean_html_text(soup: BeautifulSoup) -> str:
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    text = soup.get_text(separator=" ")
    
    return re.sub(r"\s+", " ", text).strip()


def _fetch_curated_html(urls: list[str], source_domain: str) -> tuple[list[dict], int]:
    
    records = []
    n_fetched = 0

    for url in urls:
        try:
            response = requests.get(url, headers=REQUEST_HEADERS, timeout=15)
            response.raise_for_status()
            time.sleep(DELAY_SECONDS)

            soup = BeautifulSoup(response.text, "html.parser")
            title_tag = soup.find("title")
            title = title_tag.get_text().strip() if title_tag else url
            full_text = _clean_html_text(soup)

            records.append({
                "id": url,
                "title": title,
                "abstract": full_text[:500],
                "full_text": full_text,
                "url": url,
                "source": "Curated URL",
                "source_domain": source_domain,
                "source_query": url
            })

            n_fetched += 1
            curated_logger.info(f"Successfully fetched curated {source_domain} page: {title}")

        except Exception as e:
            curated_logger.error(f"Failed to fetch curated {source_domain} URL '{url}': {e}")

    return records, n_fetched


def fetch_curated_data(
        INPUT_CSV: Path, 
        OUTPUT_RECORDS: Path,
        OUTPUT_LOGS: Path
) -> None:

    wiki_targets, external_urls = _load_curated_schedule(INPUT_CSV)

    curated_records = []
    curated_logs = []

    curated_logger.info(f"Fetching {len(wiki_targets)} unique curated Wikipedia pages.")

    wiki_records, n_wiki = _fetch_curated_wiki(wiki_targets)

    curated_records.extend(wiki_records)
    curated_logs.append({
        "source_domain": "curated_wiki",
        "items_attempted": len(wiki_targets),
        "items_fetched": n_wiki,
        "timestamp": datetime.now(IST),
    })

    if external_urls.get("arxiv"):
        curated_logger.info(f"Fetching {len(external_urls["arxiv"])} curated arxiv papers.")

        arxiv_records, n_arxiv = _fetch_curated_arxiv(external_urls["arxiv"])

        curated_records.extend(arxiv_records)
        curated_logs.append({
            "source_domain": "curated_arxiv",
            "items_attempted": len(external_urls["arxiv"]),
            "items_fetched": n_arxiv,
            "timestamp": datetime.now(IST),
        })

    for bucket in ("plato_stanford", "nobelprize"):
        urls = external_urls.get(bucket)

        if not urls:
            continue

        curated_logger.info(f"Fetching {len(urls)} curated {bucket} pages.")

        html_records, n_html = _fetch_curated_html(urls, f"curated_{bucket}")

        curated_records.extend(html_records)
        curated_logs.append({
            "source_domain": f"curated_{bucket}",
            "items_attempted": len(urls),
            "items_fetched": n_html,
            "timestamp": datetime.now(IST),
        })

    if external_urls.get("unknown"):

        curated_logger.warning(
            f"{len(external_urls["unknown"])} curated URLs did not match a known "
            f"source bucket and were skipped: {external_urls["unknown"]}"
        )

    curated_records = pl.DataFrame(curated_records) if curated_records else pl.DataFrame()
    curated_logs = pl.DataFrame(curated_logs)

    save_scraped_data(
        curated_records,
        curated_logs,
        curated_logger,
        OUTPUT_RECORDS,
        OUTPUT_LOGS
    )