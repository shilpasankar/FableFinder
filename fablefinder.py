"""
fablefinder.py

Utilities for:
1. Fetching fantasy books from Open Library and building a catalogue.
2. Deduplicating and merging with previous catalogues.
3. Finding the latest YYMMDD catalogue file.
"""

import requests
import pandas as pd
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Directory where catalogue CSVs live
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


# ---------------------------
# 1. Fetch from Open Library
# ---------------------------

def fetch_openlibrary_fantasy(
    limit_per_page: int = 500,
    max_pages: int = 10,
    published_after: Optional[int] = None,
) -> pd.DataFrame:
    """
    Fetch fantasy books from Open Library "subjects" API.

    Args:
        limit_per_page: Items per page (max 1000 per Open Library docs, but 500 is safe).
        max_pages: How many pages to fetch (rough cap).
        published_after: Only keep books with first_publish_year >= this value (if provided).

    Returns:
        DataFrame with normalized columns.
    """
    base_url = "https://openlibrary.org/subjects/fantasy.json"
    all_works = []

    for page in range(max_pages):
        offset = page * limit_per_page
        params = {
            "limit": limit_per_page,
            "offset": offset,
        }
        resp = requests.get(base_url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        works = data.get("works", [])
        if not works:
            break

        all_works.extend(works)

        # If less than limit returned, we've exhausted results
        if len(works) < limit_per_page:
            break

    records = []
    for w in all_works:
        title = w.get("title", "")
        authors = ", ".join(a.get("name", "") for a in w.get("authors", []) if a.get("name"))
        first_publish_year = w.get("first_publish_year")
        subjects = w.get("subject", []) or w.get("subjects", [])
        subjects_str = ", ".join(subjects) if isinstance(subjects, list) else str(subjects)

        work_key = w.get("key", "")  # e.g. "/works/OL12345W"
        olid = work_key.split("/")[-1] if work_key else ""

        # Skip if we want a year filter
        if published_after is not None and isinstance(first_publish_year, int):
            if first_publish_year < published_after:
                continue

        records.append(
            {
                "openlibrary_id": olid,
                "title": title,
                "authors": authors,
                "first_publish_year": first_publish_year,
                "subjects": subjects_str,
                "edition_count": w.get("edition_count"),
                "work_key": work_key,
                "cover_id": w.get("cover_id"),
            }
        )

    df = pd.DataFrame(records)
    return df


# ---------------------------
# 2. Dedupe + merge
# ---------------------------

def dedupe_catalogue(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate the catalogue.

    Primary key: openlibrary_id if present, otherwise (title, authors) fallback.
    Keep the newest (largest first_publish_year) when there’s a conflict.
    """
    df = df.copy()

    # Normalize a few text fields for better grouping
    df["title_norm"] = df["title"].fillna("").str.strip().str.lower()
    df["authors_norm"] = df["authors"].fillna("").str.strip().str.lower()

    # Make a key that uses OLID when present; fallback to title+author
    def make_key(row):
        if row.get("openlibrary_id"):
            return f"olid::{row['openlibrary_id']}"
        return f"title_author::{row['title_norm']}::{row['authors_norm']}"

    df["_dedupe_key"] = df.apply(make_key, axis=1)

    # Sort so that newer books come first
    df["_sort_year"] = df["first_publish_year"].fillna(0)

    df = df.sort_values(by=["_dedupe_key", "_sort_year"], ascending=[True, False])
    df = df.drop_duplicates(subset="_dedupe_key", keep="first")

    # Clean helper columns
    df = df.drop(columns=["title_norm", "authors_norm", "_dedupe_key", "_sort_year"], errors="ignore")

    return df.reset_index(drop=True)


def load_existing_catalogues() -> pd.DataFrame:
    """
    Load ALL existing catalogue_YYMMDD.csv files and merge them.

    Returns a possibly empty DataFrame.
    """
    pattern = re.compile(r"catalogue_(\d{6})\.csv$")
    frames: List[pd.DataFrame] = []

    for f in DATA_DIR.glob("catalogue_*.csv"):
        if not pattern.match(f.name):
            continue
        try:
            frame = pd.read_csv(f)
            frames.append(frame)
        except Exception:
            # Ignore bad files, but you could log this somewhere
            continue

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    return combined


def build_weekly_catalogue(
    published_after: Optional[int] = None,
) -> Path:
    """
    Fetch new fantasy data, merge with prior catalogues, dedupe, and write new YYMMDD file.

    Returns:
        Path to the newly written catalogue CSV.
    """
    today_code = datetime.today().strftime("%y%m%d")
    out_path = DATA_DIR / f"catalogue_{today_code}.csv"

    new_df = fetch_openlibrary_fantasy(published_after=published_after)

    existing_df = load_existing_catalogues()

    if not existing_df.empty:
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df

    combined = dedupe_catalogue(combined)

    combined.to_csv(out_path, index=False)

    return out_path


# ---------------------------
# 3. Use latest file
# ---------------------------

def get_latest_catalogue_path() -> Optional[Path]:
    """
    Find the latest catalogue_YYMMDD.csv in DATA_DIR.

    Returns:
        Path or None if no files exist.
    """
    pattern = re.compile(r"catalogue_(\d{6})\.csv$")
    files = []

    for f in DATA_DIR.glob("catalogue_*.csv"):
        m = pattern.match(f.name)
        if m:
            files.append((m.group(1), f))

    if not files:
        return None

    # Sort by date code string (YYMMDD) – lexicographic matches chronological here
    latest_code, latest_file = sorted(files, key=lambda x: x[0])[-1]
    return latest_file


def load_latest_catalogue() -> pd.DataFrame:
    """
    Load the latest catalogue CSV, or raise a clear error if missing.
    """
    latest = get_latest_catalogue_path()
    if latest is None:
        raise FileNotFoundError(
            "No catalogue_YYMMDD.csv files found in data/. "
            "Run build_weekly_catalogue() first."
        )
    df = pd.read_csv(latest)
    return df


# ---------------------------
# CLI helper (optional)
# ---------------------------

if __name__ == "__main__":
    """
    Example CLI usage:

    python fablefinder.py          # builds catalogue for all years
    python fablefinder.py 2000     # only keep books published >= 2000
    """
    import sys

    year_filter = None
    if len(sys.argv) > 1:
        try:
            year_filter = int(sys.argv[1])
        except ValueError:
            print("Invalid year; ignoring filter.")

    out = build_weekly_catalogue(published_after=year_filter)
    print(f"✅ Wrote catalogue: {out}")
