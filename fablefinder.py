"""
fablefinder.py

Utilities for:
1. Fetching fantasy books from Open Library and building a catalogue.
2. Deduplicating and merging with previous catalogues.
3. Finding the latest YYMMDD catalogue file.
"""
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

import csv
import json
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
# ---------------------------
# CMU Book Summary Dataset integration
# ---------------------------

def load_cmu_books(path: str = "/kaggle/input/cmu-book-summary-dataset/booksummaries.txt"):
    """
    Load the CMU Book Summary dataset (tab-separated).

    Each row:
    [wiki_id, freebase_id, title, author, pub_date, genres_json, summary]
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, dialect="excel-tab")
        for row in tqdm(reader, desc="Loading CMU summaries"):
            if not row:
                continue
            rows.append(row)
    return rows


def build_cmu_df(cmu_rows) -> pd.DataFrame:
    """
    Turn raw CMU rows into a DataFrame with clean columns:
    wiki_id, freebase_id, title, author, pub_date, genres_raw, genres_list, summary
    """
    processed = []

    for row in cmu_rows:
        if len(row) < 7:
            continue

        wiki_id = row[0]
        freebase_id = row[1]
        title = row[2]
        author = row[3]
        pub_date = row[4]
        genres_raw = row[5]
        summary = row[6]

        # genres_raw is a dict-like string; normalise to a simple list of genre names
        genres_list = []
        if genres_raw:
            try:
                # CMU uses single quotes, invalid JSON → fix them
                genre_dict = json.loads(genres_raw.replace("'", '"'))
                if isinstance(genre_dict, dict):
                    genres_list = list(genre_dict.keys())
            except Exception:
                # if parsing fails, leave genres_list empty but keep raw string
                genres_list = []

        processed.append(
            {
                "wiki_id": wiki_id,
                "freebase_id": freebase_id,
                "cmu_title": title,
                "cmu_author": author,
                "cmu_pub_date": pub_date,
                "cmu_genres_raw": genres_raw,
                "cmu_genres_list": "; ".join(genres_list),
                "cmu_summary": summary,
            }
        )

    return pd.DataFrame(processed)


def build_weekly_catalogue(
    published_after: Optional[int] = None,
    cmu_path: str = "/kaggle/input/cmu-book-summary-dataset/booksummaries.txt",
) -> Path:
    """
    Fetch new fantasy data from Open Library, merge with prior catalogues,
    dedupe, enrich with CMU Book Summary dataset, and write new YYMMDD file.

    Returns:
        Path to the newly written catalogue CSV.
    """
    today_code = datetime.today().strftime("%y%m%d")
    out_path = DATA_DIR / f"catalogue_{today_code}.csv"

    # 1. Fetch fresh Open Library fantasy data
    new_df = fetch_openlibrary_fantasy(published_after=published_after)

    # 2. Load existing catalogue (if any)
    existing_df = load_existing_catalogues()

    if not existing_df.empty:
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df

    # 3. Dedupe
    combined = dedupe_catalogue(combined)

    # 4. Load and build CMU DF
    try:
        cmu_rows = load_cmu_books(cmu_path)
        cmu_df = build_cmu_df(cmu_rows)
        combined = enrich_with_cmu(combined, cmu_df)
    except FileNotFoundError:
        print(f"Warning: CMU dataset not found at {cmu_path}. Skipping CMU enrichment.")
    except Exception as e:
        print(f"Warning: failed to enrich with CMU dataset: {e}")

    # 5. Basic schema completion (optional: ensure summary/genre/type exist)
    if "type" not in combined.columns:
        combined["type"] = "fiction"  # everything here is fantasy
    if "genre" not in combined.columns:
        combined["genre"] = "Fantasy"
    if "summary" not in combined.columns:
        combined["summary"] = pd.NA

    combined.to_csv(out_path, index=False)
    return out_path

# ---------------------------
# 2b. Normalise / complete schema
# ---------------------------

EXPECTED_COLUMNS = [
    "openlibrary_id",
    "title",
    "authors",
    "first_publish_year",
    "subjects",
    "edition_count",
    "work_key",
    "cover_id",
    # "semantic" fields used by the app:
    "type",
    "genre",
    "mood",
    "pace",
    "tropes",
    "hero_heroine",
    "devices",
    "page_count",
    "in_series",
    "series_name",
    "series_length",
    "owned",
    "to_read",
]


def infer_genre_from_subjects(subjects_str: str) -> str:
    """
    Very rough heuristic to turn subjects into a coarse 'genre' string.
    This runs on each row; it's okay if it's not perfect – you can overwrite later.
    """
    if not isinstance(subjects_str, str):
        return "Fantasy"

    s = subjects_str.lower()

    if "urban fantasy" in s:
        return "Urban Fantasy"
    if "young adult" in s or "ya fiction" in s:
        return "YA Fantasy"
    if "children" in s or "juvenile fiction" in s:
        return "Middle Grade / Children"
    if "grimdark" in s or "dark fantasy" in s:
        return "Grimdark"
    if "cozy" in s or "comfort read" in s:
        return "Cozy Fantasy"
    if "mythology" in s or "folklore" in s:
        return "Mythic Fantasy"

    # Fallback
    return "Fantasy"


def normalise_catalogue_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the catalogue has all columns the Streamlit app expects,
    with sensible defaults where we can infer them.
    """
    df = df.copy()

    # Make sure all EXPECTED_COLUMNS exist
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    # TYPE: default all fantasy books to 'fiction' unless already set
    if "type" in df.columns:
        df["type"] = df["type"].fillna("fiction")

    # GENRE: infer from subjects if missing/empty
    if "genre" in df.columns:
        mask_empty = df["genre"].isna() | (df["genre"].astype(str).str.strip() == "")
        df.loc[mask_empty, "genre"] = df.loc[mask_empty, "subjects"].apply(infer_genre_from_subjects)

    # BOOLS: in_series, owned, to_read – normalise to True/False/NA
    for bool_col in ["in_series", "owned", "to_read"]:
        if bool_col in df.columns:
            df[bool_col] = df[bool_col].map(
                lambda x: True if x is True or str(x).lower() in ["true", "1", "yes", "y"]
                else False if x is False or str(x).lower() in ["false", "0", "no", "n"]
                else pd.NA
            )

    # SERIES LENGTH: if missing but we know it's not series, set to 1 (standalone)
    if "series_length" in df.columns and "in_series" in df.columns:
        # if series_length is NaN and in_series is False/NA, call it 1 (standalone)
        mask = df["series_length"].isna() & (df["in_series"] != True)
        df.loc[mask, "series_length"] = 1

    # PAGE COUNT: leave as-is if present; otherwise stays NA until enriched elsewhere

    return df
def normalise_str(s):
    if not isinstance(s, str):
        return ""
    return (
        s.lower()
        .strip()
        .replace(".", "")
        .replace(",", "")
        .replace(":", "")
        .replace(";", "")
        .replace("!", "")
        .replace("?", "")
    )


def enrich_with_cmu(base_df: pd.DataFrame, cmu_df: pd.DataFrame) -> pd.DataFrame:
    """
    Match Open Library catalogue to CMU Book Summary dataset by (title, author)
    and enrich with:
      - genre (from cmu_genres_list)
      - summary (from cmu_summary)

    Notes:
      - We do a simple normalised exact match on title + author.
      - If a row already has 'genre' or 'summary', CMU only fills missing values.
    """
    df = base_df.copy()

    # normalised keys for join
    df["title_norm"] = df["title"].fillna("").apply(normalise_str)
    df["author_norm"] = df["authors"].fillna("").apply(normalise_str)

    cmu = cmu_df.copy()
    cmu["cmu_title_norm"] = cmu["cmu_title"].fillna("").apply(normalise_str)
    cmu["cmu_author_norm"] = cmu["cmu_author"].fillna("").apply(normalise_str)

    # build small key to join on
    df["match_key"] = df["title_norm"] + "::" + df["author_norm"]
    cmu["match_key"] = cmu["cmu_title_norm"] + "::" + cmu["cmu_author_norm"]

    # drop duplicates on CMU side (keep first)
    cmu_small = cmu[
        ["match_key", "cmu_genres_list", "cmu_summary"]
    ].drop_duplicates(subset="match_key", keep="first")

    merged = df.merge(cmu_small, on="match_key", how="left")

    # ensure genre + summary columns exist
    if "genre" not in merged.columns:
        merged["genre"] = pd.NA
    if "summary" not in merged.columns:
        merged["summary"] = pd.NA

    # fill missing genre/summary from CMU
    mask_genre_missing = merged["genre"].isna() | (merged["genre"].astype(str).str.strip() == "")
    merged.loc[mask_genre_missing, "genre"] = merged["cmu_genres_list"]

    mask_summary_missing = merged["summary"].isna() | (merged["summary"].astype(str).str.strip() == "")
    merged.loc[mask_summary_missing, "summary"] = merged["cmu_summary"]

    # clean up helper columns
    merged = merged.drop(
        columns=[
            "title_norm",
            "author_norm",
            "match_key",
            "cmu_genres_list",
            "cmu_summary",
        ],
        errors="ignore",
    )

    return merged


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
