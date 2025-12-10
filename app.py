import streamlit as st
import pandas as pd

from fablefinder import load_latest_catalogue, get_latest_catalogue_path


# ---------- helpers for token-like columns ----------

def explode_tokens(series: pd.Series, delimiters=(",",";")):
    """
    Turn a column of strings like "cozy; whimsical, low stakes"
    into a flat set of unique tokens.
    """
    tokens = set()
    for val in series.dropna().astype(str):
        s = val
        for d in delimiters:
            s = s.replace(d, ";")
        for part in s.split(";"):
            part = part.strip()
            if part:
                tokens.add(part)
    return sorted(tokens)


def value_contains_any(cell: str, candidates) -> bool:
    """
    Returns True if any candidate appears as a token/substring
    in the given cell value.
    """
    if not candidates:
        return False
    if pd.isna(cell):
        return False
    text = str(cell).lower()
    return any(c.lower() in text for c in candidates)


# ---------- filter logic ----------

def apply_filters(
    df: pd.DataFrame,
    search: str,
    year_min: int,
    year_max: int,
    include_moods,
    exclude_moods,
    include_pace,
    exclude_pace,
    include_type,
    exclude_type,
    include_genre,
    exclude_genre,
    include_tropes,
    exclude_tropes,
    include_heroes,
    exclude_heroes,
    include_devices,
    exclude_devices,
    page_buckets,
    ownership_mode: str,
) -> pd.DataFrame:
    filtered = df.copy()

    # --- basic year filter ---
    if "first_publish_year" in filtered.columns:
        fp = filtered["first_publish_year"].fillna(0)
        filtered = filtered[(fp >= year_min) & (fp <= year_max)]

    # --- ownership / series filter ---
    if ownership_mode == "Books I own":
        if "owned" in filtered.columns:
            filtered = filtered[filtered["owned"] == True]
        else:
            st.warning("Ownership filter selected but 'owned' column not found.")
    elif ownership_mode == "Books in my to-read pile":
        if "to_read" in filtered.columns:
            filtered = filtered[filtered["to_read"] == True]
        else:
            st.warning("To-read filter selected but 'to_read' column not found.")
    elif ownership_mode == "Books not in a series":
        if "in_series" in filtered.columns:
            filtered = filtered[(filtered["in_series"] == False) | (filtered["in_series"].isna())]
        elif "series_name" in filtered.columns:
            filtered = filtered[filtered["series_name"].isna() | (filtered["series_name"].astype(str).str.strip() == "")]
        else:
            st.warning("Series filter selected but neither 'in_series' nor 'series_name' columns found.")

    # --- page count bucket filter ---
    if "page_count" in filtered.columns and page_buckets:
        pc = filtered["page_count"].fillna(0).astype(int)

        mask = pd.Series(False, index=filtered.index)
        if "<300" in page_buckets:
            mask |= pc < 300
        if "300–499" in page_buckets:
            mask |= (pc >= 300) & (pc <= 499)
        if "500+" in page_buckets:
            mask |= pc >= 500

        filtered = filtered[mask]

    # --- helper to include/exclude on token-like columns ---
    def include_exclude(column_name: str, include_vals, exclude_vals):
        nonlocal filtered
        if column_name not in filtered.columns:
            return

        col = filtered[column_name].astype(str)

        if include_vals:
            inc_mask = col.apply(lambda x: value_contains_any(x, include_vals))
            filtered = filtered[inc_mask]

        if exclude_vals:
            exc_mask = col.apply(lambda x: value_contains_any(x, exclude_vals))
            filtered = filtered[~exc_mask]

    include_exclude("mood", include_moods, exclude_moods)
    include_exclude("pace", include_pace, exclude_pace)
    include_exclude("type", include_type, exclude_type)
    include_exclude("genre", include_genre, exclude_genre)
    include_exclude("tropes", include_tropes, exclude_tropes)
    include_exclude("hero_heroine", include_heroes, exclude_heroes)
    include_exclude("devices", include_devices, exclude_devices)

    # --- optional free-text search ---
    if search:
        s = search.lower()

        def col_contains(col_name):
            if col_name not in filtered.columns:
                return pd.Series(False, index=filtered.index)
            return filtered[col_name].astype(str).str.lower().str.contains(s)

        mask = (
            col_contains("title")
            | col_contains("authors")
            | col_contains("genre")
            | col_contains("tropes")
            | col_contains("hero_heroine")
            | col_contains("devices")
            | col_contains("subjects")
        )
        filtered = filtered[mask]

    return filtered


# ---------- UI ----------

def main():
    st.set_page_config(
        page_title="FableFinder",
        page_icon="📚",
        layout="wide",
    )

    st.title("FableFinder 🧙📚")
    st.caption("Filter your fantasy catalogue by mood, pace, tropes, devices, and more.")

    latest_path = get_latest_catalogue_path()
    if latest_path is None:
        st.error(
            "No catalogue files found in `data/`.\n\n"
            "Run `python fablefinder.py` to create `catalogue_YYMMDD.csv` first."
        )
        st.stop()

    st.info(f"Using latest catalogue: `{latest_path.name}`")

    try:
        df = load_latest_catalogue()
    except Exception as e:
        st.error(f"Could not load latest catalogue: {e}")
        st.stop()

    # Ensure basic columns exist
    for col in ["title", "authors", "first_publish_year"]:
        if col not in df.columns:
            df[col] = ""

    # Sidebar filters
    with st.sidebar:
        st.header("🔍 Filters")

        # Ownership / series selector
        ownership_mode = st.radio(
            "Search only:",
            options=[
                "All books",
                "Books I own",
                "Books in my to-read pile",
                "Books not in a series",
            ],
            index=0,
        )

        # Year slider
        if df["first_publish_year"].notna().any():
            min_year = int(df["first_publish_year"].min())
            max_year = int(df["first_publish_year"].max())
        else:
            min_year, max_year = 1800, 2025

        year_min, year_max = st.slider(
            "First publish year range",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year),
        )

        # Page count buckets
        bucket_options = ["<300", "300–499", "500+",]
        page_buckets = st.multiselect(
            "Page count",
            options=bucket_options,
            help="Apply 0, 1, or multiple page count buckets.",
        )

        st.markdown("---")
        st.caption("Advanced filters:")

        # Mood
        mood_vals = explode_tokens(df["mood"]) if "mood" in df.columns else []
        include_moods = st.multiselect("Include mood", options=mood_vals, key="inc_mood")
        exclude_moods = st.multiselect("Exclude mood", options=mood_vals, key="exc_mood")

        # Pace
        pace_vals = ["slow", "medium", "fast"]
        include_pace = st.multiselect("Include pace", options=pace_vals, key="inc_pace")
        exclude_pace = st.multiselect("Exclude pace", options=pace_vals, key="exc_pace")

        # Type
        type_vals = ["fiction", "non-fiction"]
        include_type = st.multiselect("Include type", options=type_vals, key="inc_type")
        exclude_type = st.multiselect("Exclude type", options=type_vals, key="exc_type")

        # Genre
        genre_vals = explode_tokens(df["genre"]) if "genre" in df.columns else []
        include_genre = st.multiselect("Include genre", options=genre_vals, key="inc_genre")
        exclude_genre = st.multiselect("Exclude genre", options=genre_vals, key="exc_genre")

        # Tropes
        trope_vals = explode_tokens(df["tropes"]) if "tropes" in df.columns else []
        include_tropes = st.multiselect("Include tropes", options=trope_vals, key="inc_tropes")
        exclude_tropes = st.multiselect("Exclude tropes", options=trope_vals, key="exc_tropes")

        # Hero / heroine
        hero_vals = explode_tokens(df["hero_heroine"]) if "hero_heroine" in df.columns else []
        include_heroes = st.multiselect("Include hero/heroine", options=hero_vals, key="inc_heroes")
        exclude_heroes = st.multiselect("Exclude hero/heroine", options=hero_vals, key="exc_heroes")

        # Devices
        device_vals = explode_tokens(df["devices"]) if "devices" in df.columns else []
        include_devices = st.multiselect("Include literary devices", options=device_vals, key="inc_devices")
        exclude_devices = st.multiselect("Exclude literary devices", options=device_vals, key="exc_devices")

        st.markdown("---")
        search = st.text_input(
            "Free text search (optional)",
            placeholder="title, author, trope, device, etc.",
            help="Leave empty to filter only by the controls above.",
        )

    # Apply filters (search is allowed to be empty)
    filtered = apply_filters(
        df=df,
        search=search,
        year_min=year_min,
        year_max=year_max,
        include_moods=include_moods,
        exclude_moods=exclude_moods,
        include_pace=include_pace,
        exclude_pace=exclude_pace,
        include_type=include_type,
        exclude_type=exclude_type,
        include_genre=include_genre,
        exclude_genre=exclude_genre,
        include_tropes=include_tropes,
        exclude_tropes=exclude_tropes,
        include_heroes=include_heroes,
        exclude_heroes=exclude_heroes,
        include_devices=include_devices,
        exclude_devices=exclude_devices,
        page_buckets=page_buckets,
        ownership_mode=ownership_mode,
    )

    st.subheader(f"Results ({len(filtered)})")

    if filtered.empty:
        st.warning("No books match your filters yet. Try relaxing one or two constraints.")
        return

    # Layout: table + details
    left, right = st.columns([1.5, 1])

    show_cols = [c for c in [
        "title",
        "authors",
        "type",
        "genre",
        "mood",
        "pace",
        "first_publish_year",
        "page_count",
    ] if c in filtered.columns]

    with left:
        st.dataframe(
            filtered[show_cols],
            use_container_width=True,
            hide_index=True,
        )

    with right:
        st.subheader("Book details")

        options = [
            f"{row.get('title', '')} — {row.get('authors','')} "
            f"({int(row['first_publish_year']) if pd.notna(row.get('first_publish_year')) else 'n/a'})"
            for _, row in filtered.iterrows()
        ]
        selected = st.selectbox("Pick a book", options)

        idx = options.index(selected)
        book = filtered.iloc[idx]

        st.markdown(f"### {book.get('title','')}")
        st.markdown(f"*by {book.get('authors','')}*")

        if pd.notna(book.get("first_publish_year")):
            st.markdown(f"**First published:** {int(book['first_publish_year'])}")

        if "type" in book:
            st.markdown(f"**Type:** {book.get('type', '')}")
        if "genre" in book:
            st.markdown(f"**Genre:** {book.get('genre', '')}")
        if "mood" in book:
            st.markdown(f"**Mood:** {book.get('mood', '')}")
        if "pace" in book:
            st.markdown(f"**Pace:** {book.get('pace', '')}")
        if "tropes" in book:
            st.markdown(f"**Tropes:** {book.get('tropes', '')}")
        if "hero_heroine" in book:
            st.markdown(f"**Hero / Heroine:** {book.get('hero_heroine', '')}")
        if "devices" in book:
            st.markdown(f"**Literary devices:** {book.get('devices', '')}")

        if "page_count" in book and pd.notna(book.get("page_count")):
            st.markdown(f"**Page count:** {int(book['page_count'])}")

        # Ownership info if present
        owned = book.get("owned", None)
        to_read = book.get("to_read", None)
        in_series = book.get("in_series", None)

        meta_bits = []
        if isinstance(owned, bool):
            meta_bits.append("✅ Owned" if owned else "Not owned")
        if isinstance(to_read, bool):
            meta_bits.append("📌 In to-read pile" if to_read else "Not in to-read pile")
        if isinstance(in_series, bool):
            meta_bits.append("📚 Part of a series" if in_series else "Standalone")

        if meta_bits:
            st.markdown("**Status:** " + " · ".join(meta_bits))

        # Cover if available
        cover_id = book.get("cover_id", None)
        if pd.notna(cover_id):
            cover_url = f"https://covers.openlibrary.org/b/id/{int(cover_id)}-L.jpg"
            st.image(cover_url, caption="Open Library cover", use_container_width=True)


if __name__ == "__main__":
    main()
