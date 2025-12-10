import streamlit as st
import pandas as pd

from fablefinder import load_latest_catalogue, get_latest_catalogue_path


# ---------- defaults for filters (so they’re never empty) ----------

DEFAULT_GENRES = [
    "Epic Fantasy",
    "Urban Fantasy",
    "Grimdark",
    "Cozy Fantasy",
    "YA Fantasy",
    "Mythic",
    "Portal Fantasy",
]

DEFAULT_PACE = ["slow", "medium", "fast"]

DEFAULT_HERO_TYPES = [
    "female lead",
    "male lead",
    "non-binary lead",
    "ensemble cast",
    "anti-hero",
    "reluctant hero",
]

DEFAULT_DEVICES = [
    "multiple POV",
    "unreliable narrator",
    "non-linear timeline",
    "frame narrative",
    "epistolary",
    "breaking the fourth wall",
]


# ---------- helpers for token-like columns ----------

def explode_tokens(series: pd.Series, delimiters=(",",";")):
    """
    Turn a column of strings like "cozy; whimsical, low stakes"
    into a flat sorted list of unique tokens.
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
    series_mode: str,
    series_length_filters,
) -> pd.DataFrame:
    filtered = df.copy()

    # --- basic year filter ---
    if "first_publish_year" in filtered.columns:
        fp = filtered["first_publish_year"].fillna(0)
        filtered = filtered[(fp >= year_min) & (fp <= year_max)]

    # --- series mode filter (search only) ---
    # modes: "All books", "Books not in a series", "Books in a series"
    if series_mode == "Books not in a series":
        if "in_series" in filtered.columns:
            filtered = filtered[(filtered["in_series"] == False) | (filtered["in_series"].isna())]
        elif "series_name" in filtered.columns:
            filtered = filtered[
                filtered["series_name"].isna()
                | (filtered["series_name"].astype(str).str.strip() == "")
            ]
    elif series_mode == "Books in a series":
        if "in_series" in filtered.columns:
            filtered = filtered[filtered["in_series"] == True]
        elif "series_name" in filtered.columns:
            filtered = filtered[
                filtered["series_name"].notna()
                & (filtered["series_name"].astype(str).str.strip() != "")
            ]

    # --- series length filter (if we have a 'series_length' column) ---
    # buckets: "Standalone (1)", "Duology (2)", "Trilogy (3)", "Short series (4–6)", "Long series (7+)"
    if "series_length" in filtered.columns and series_length_filters:
        sl = filtered["series_length"].fillna(1).astype(int)  # assume NaN = standalone

        mask = pd.Series(False, index=filtered.index)
        if "Standalone (1)" in series_length_filters:
            mask |= (sl == 1)
        if "Duology (2)" in series_length_filters:
            mask |= (sl == 2)
        if "Trilogy (3)" in series_length_filters:
            mask |= (sl == 3)
        if "Short series (4–6)" in series_length_filters:
            mask |= (sl >= 4) & (sl <= 6)
        if "Long series (7+)" in series_length_filters:
            mask |= (sl >= 7)

        filtered = filtered[mask]

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


# ---------- grid render helpers ----------

def render_book_grid(df: pd.DataFrame, max_books: int = 50):
    """
    Show books as small cards in a grid (5 per row).
    """
    display_df = df.head(max_books)

    if display_df.empty:
        st.write("No books to show.")
        return

    # iterate in chunks of 5
    for start in range(0, len(display_df), 5):
        row_slice = display_df.iloc[start:start+5]
        cols = st.columns(len(row_slice))

        for col, (_, book) in zip(cols, row_slice.iterrows()):
            with col:
                # cover
                cover_id = book.get("cover_id", None)
                if pd.notna(cover_id):
                    cover_url = f"https://covers.openlibrary.org/b/id/{int(cover_id)}-M.jpg"
                    # smaller image
                    st.image(cover_url, use_container_width=False, width=110)
                else:
                    st.empty()

                title = str(book.get("title", ""))
                author = str(book.get("authors", ""))[:60]
                year = book.get("first_publish_year", "")

                st.markdown(f"**{title[:60]}{'…' if len(title) > 60 else ''}**")
                if author:
                    st.caption(author)
                if pd.notna(year):
                    st.caption(f"Year: {int(year)}")


# ---------- main UI ----------

def main():
    st.set_page_config(
        page_title="FableFinder",
        page_icon="📚",
        layout="wide",
    )

    st.title("FableFinder 🧙📚")
    st.caption("Filter your fantasy catalogue by mood, pace, series length, tropes, and more.")

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

    # --- build option lists: default + data-driven ---
    mood_options = explode_tokens(df["mood"]) if "mood" in df.columns else []
    pace_options = list(sorted(set(DEFAULT_PACE +
                                  (explode_tokens(df["pace"]) if "pace" in df.columns else []))))
    type_options = ["fiction", "non-fiction"]
    genre_options = list(sorted(set(DEFAULT_GENRES +
                                   (explode_tokens(df["genre"]) if "genre" in df.columns else []))))
    trope_options = explode_tokens(df["tropes"]) if "tropes" in df.columns else []
    hero_options = list(sorted(set(DEFAULT_HERO_TYPES +
                                  (explode_tokens(df["hero_heroine"]) if "hero_heroine" in df.columns else []))))
    device_options = list(sorted(set(DEFAULT_DEVICES +
                                    (explode_tokens(df["devices"]) if "devices" in df.columns else []))))

    # Sidebar filters
    with st.sidebar:
        st.header("🔍 Filters")

        # "Search only" = series mode
        series_mode = st.radio(
            "Search only:",
            options=[
                "All books",
                "Books not in a series",
                "Books in a series",
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
            "First publish year",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year),
        )

        # Page count buckets
        bucket_options = ["<300", "300–499", "500+"]
        page_buckets = st.multiselect(
            "Page count",
            options=bucket_options,
        )

        # Series length filter
        st.markdown("**Series length**")
        series_length_filters = st.multiselect(
            "Number of books in series",
            options=[
                "Standalone (1)",
                "Duology (2)",
                "Trilogy (3)",
                "Short series (4–6)",
                "Long series (7+)",
            ],
            help="Uses `series_length` column if available.",
        )

        st.markdown("---")
        st.caption("Mood, pace, type, genre, tropes, hero, devices")

        # Mood include/exclude side by side
        st.markdown("**Mood**")
        c1, c2 = st.columns(2)
        with c1:
            include_moods = st.multiselect("Include", options=mood_options, key="inc_mood")
        with c2:
            exclude_moods = st.multiselect("Exclude", options=mood_options, key="exc_mood")

        # Pace include/exclude
        st.markdown("**Pace**")
        c1, c2 = st.columns(2)
        with c1:
            include_pace = st.multiselect("Include", options=pace_options, key="inc_pace")
        with c2:
            exclude_pace = st.multiselect("Exclude", options=pace_options, key="exc_pace")

        # Type include/exclude
        st.markdown("**Type**")
        c1, c2 = st.columns(2)
        with c1:
            include_type = st.multiselect("Include", options=type_options, key="inc_type")
        with c2:
            exclude_type = st.multiselect("Exclude", options=type_options, key="exc_type")

        # Genre include/exclude
        st.markdown("**Genre**")
        c1, c2 = st.columns(2)
        with c1:
            include_genre = st.multiselect("Include", options=genre_options, key="inc_genre")
        with c2:
            exclude_genre = st.multiselect("Exclude", options=genre_options, key="exc_genre")

        # Tropes include/exclude
        st.markdown("**Tropes**")
        c1, c2 = st.columns(2)
        with c1:
            include_tropes = st.multiselect("Include", options=trope_options, key="inc_tropes")
        with c2:
            exclude_tropes = st.multiselect("Exclude", options=trope_options, key="exc_tropes")

        # Hero / heroine include/exclude
        st.markdown("**Hero / Heroine**")
        c1, c2 = st.columns(2)
        with c1:
            include_heroes = st.multiselect("Include", options=hero_options, key="inc_heroes")
        with c2:
            exclude_heroes = st.multiselect("Exclude", options=hero_options, key="exc_heroes")

        # Literary devices include/exclude
        st.markdown("**Literary devices**")
        c1, c2 = st.columns(2)
        with c1:
            include_devices = st.multiselect("Include", options=device_options, key="inc_devices")
        with c2:
            exclude_devices = st.multiselect("Exclude", options=device_options, key="exc_devices")

        st.markdown("---")
        search = st.text_input(
            "Free text search (optional)",
            placeholder="title, author, trope, device, etc.",
            help="Leave empty to use only the filters.",
        )

    # Apply filters (search can be empty)
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
        series_mode=series_mode,
        series_length_filters=series_length_filters,
    )

    st.subheader(f"Results ({len(filtered)})")

    if filtered.empty:
        st.warning("No books match your filters yet. Try relaxing one or two constraints.")
        return

    # Grid of books at the top
    st.markdown("### Books")
    render_book_grid(filtered)

    # Detailed pane below
    st.markdown("---")
    st.markdown("### Details")

    options = [
        f"{row.get('title', '')} — {row.get('authors','')} "
        f"({int(row['first_publish_year']) if pd.notna(row.get('first_publish_year')) else 'n/a'})"
        for _, row in filtered.iterrows()
    ]
    selected = st.selectbox("Pick a book for full details", options)

    idx = options.index(selected)
    book = filtered.iloc[idx]

    left, right = st.columns([1, 2])

    with left:
        cover_id = book.get("cover_id", None)
        if pd.notna(cover_id):
            cover_url = f"https://covers.openlibrary.org/b/id/{int(cover_id)}-M.jpg"
            st.image(cover_url, caption="Cover", use_container_width=False, width=130)

    with right:
        st.markdown(f"#### {book.get('title','')}")
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

        if "series_length" in book and pd.notna(book.get("series_length")):
            st.markdown(f"**Series length:** {int(book['series_length'])} book(s)")

        if "series_name" in book and pd.notna(book.get("series_name")):
            st.markdown(f"**Series name:** {book.get('series_name', '')}")


if __name__ == "__main__":
    main()
