import streamlit as st
import pandas as pd

from fablefinder import load_latest_catalogue, get_latest_catalogue_path


# ---------- fixed option sets (so filters never vanish) ----------

MOOD_OPTIONS = [
    "cozy",
    "whimsical",
    "dark",
    "grim",
    "hopeful",
    "political",
    "romantic",
]

PACE_OPTIONS = ["slow", "medium", "fast"]

TYPE_OPTIONS = ["fiction", "non-fiction"]

GENRE_OPTIONS = [
    "Epic Fantasy",
    "Urban Fantasy",
    "Grimdark",
    "Cozy Fantasy",
    "YA Fantasy",
    "Mythic",
    "Portal Fantasy",
]

TROPE_OPTIONS = [
    "found family",
    "chosen one",
    "enemies to lovers",
    "slow burn",
    "heist",
    "dragons",
    "magic school",
]

HERO_OPTIONS = [
    "female lead",
    "male lead",
    "non-binary lead",
    "ensemble cast",
    "anti-hero",
    "reluctant hero",
]

DEVICE_OPTIONS = [
    "multiple POV",
    "unreliable narrator",
    "non-linear timeline",
    "frame narrative",
    "epistolary",
    "breaking the fourth wall",
]


# ---------- helpers ----------

def value_contains_any(cell: str, candidates) -> bool:
    """True if any candidate appears (substring) in the cell."""
    if not candidates:
        return False
    if pd.isna(cell):
        return False
    text = str(cell).lower()
    return any(c.lower() in text for c in candidates)


def safe_include_exclude(df, column_name, include_vals, exclude_vals, debug_messages, label):
    """
    Apply include/exclude filters on df[column_name] if column exists.
    If not, do nothing (and explain in debug).
    """
    if not include_vals and not exclude_vals:
        return df  # nothing selected

    if column_name not in df.columns:
        debug_messages.append(f"{label}: column '{column_name}' missing – filter not applied.")
        return df

    col = df[column_name].astype(str)
    before = len(df)

    # include
    if include_vals:
        inc_mask = col.apply(lambda x: value_contains_any(x, include_vals))
        df = df[inc_mask]

    # exclude
    if exclude_vals and not df.empty:
        col = df[column_name].astype(str)
        exc_mask = col.apply(lambda x: value_contains_any(x, exclude_vals))
        df = df[~exc_mask]

    debug_messages.append(
        f"{label}: include={include_vals or '[]'}, exclude={exclude_vals or '[]'} -> {len(df)} (was {before})"
    )
    return df


# ---------- filtering logic ----------

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
):
    filtered = df.copy()
    debug_messages = []

    debug_messages.append(f"Start: {len(filtered)} books")

    # year range
    if "first_publish_year" in filtered.columns:
        fp = filtered["first_publish_year"].fillna(0)
        before = len(filtered)
        filtered = filtered[(fp >= year_min) & (fp <= year_max)]
        debug_messages.append(f"Year {year_min}-{year_max}: {len(filtered)} (was {before})")
    else:
        debug_messages.append("No 'first_publish_year' column – year filter not applied.")

    # series mode
    if series_mode == "Books not in a series":
        if "in_series" in filtered.columns:
            before = len(filtered)
            filtered = filtered[(filtered["in_series"] == False) | (filtered["in_series"].isna())]
            debug_messages.append(f"'Not in series' -> {len(filtered)} (was {before})")
        elif "series_name" in filtered.columns:
            before = len(filtered)
            filtered = filtered[
                filtered["series_name"].isna()
                | (filtered["series_name"].astype(str).str.strip() == "")
            ]
            debug_messages.append(f"'Not in series' via series_name -> {len(filtered)} (was {before})")
        else:
            debug_messages.append("Series mode 'not in series' selected but no series column found.")
    elif series_mode == "Books in a series":
        if "in_series" in filtered.columns:
            before = len(filtered)
            filtered = filtered[filtered["in_series"] == True]
            debug_messages.append(f"'In series' -> {len(filtered)} (was {before})")
        elif "series_name" in filtered.columns:
            before = len(filtered)
            filtered = filtered[
                filtered["series_name"].notna()
                & (filtered["series_name"].astype(str).str.strip() != "")
            ]
            debug_messages.append(f"'In series' via series_name -> {len(filtered)} (was {before})")
        else:
            debug_messages.append("Series mode 'in series' selected but no series column found.")

    # series length
    if series_length_filters:
        if "series_length" in filtered.columns:
            sl = filtered["series_length"].fillna(1).astype(int)
            before = len(filtered)

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
            debug_messages.append(f"Series length {series_length_filters}: {len(filtered)} (was {before})")
        else:
            debug_messages.append("Series length filters selected but no 'series_length' column found.")

    # page count
    if page_buckets:
        if "page_count" in filtered.columns:
            pc = filtered["page_count"].fillna(0).astype(int)
            before = len(filtered)

            mask = pd.Series(False, index=filtered.index)
            if "<300" in page_buckets:
                mask |= pc < 300
            if "300–499" in page_buckets:
                mask |= (pc >= 300) & (pc <= 499)
            if "500+" in page_buckets:
                mask |= pc >= 500

            filtered = filtered[mask]
            debug_messages.append(f"Page count {page_buckets}: {len(filtered)} (was {before})")
        else:
            debug_messages.append("Page count filters selected but no 'page_count' column found.")

    # include/exclude token-ish columns
    filtered = safe_include_exclude(
        filtered, "mood", include_moods, exclude_moods, debug_messages, "Mood"
    )
    filtered = safe_include_exclude(
        filtered, "pace", include_pace, exclude_pace, debug_messages, "Pace"
    )
    filtered = safe_include_exclude(
        filtered, "type", include_type, exclude_type, debug_messages, "Type"
    )
    filtered = safe_include_exclude(
        filtered, "genre", include_genre, exclude_genre, debug_messages, "Genre"
    )
    filtered = safe_include_exclude(
        filtered, "tropes", include_tropes, exclude_tropes, debug_messages, "Tropes"
    )
    filtered = safe_include_exclude(
        filtered, "hero_heroine", include_heroes, exclude_heroes, debug_messages, "Hero/Heroine"
    )
    filtered = safe_include_exclude(
        filtered, "devices", include_devices, exclude_devices, debug_messages, "Devices"
    )

    # text search (optional)
    if search:
        s = search.lower()
        before = len(filtered)

        def col_contains(col_name):
            if col_name not in filtered.columns:
                return pd.Series(False, index=filtered.index)
            return filtered[col_name].astype(str).str.lower().str.contains(s)

        mask = (
            col_contains("title")
            | col_contains("authors")
            | col_contains("genre")
            | col_contains("tropes")
            | col_contains("subjects")
        )
        filtered = filtered[mask]
        debug_messages.append(f"Search '{search}': {len(filtered)} (was {before})")
    else:
        debug_messages.append("No text search applied.")

    return filtered, debug_messages


# ---------- grid render ----------

def render_book_grid(df: pd.DataFrame, max_books: int = 50):
    """Show books as small cards in a grid (5 per row)."""
    display_df = df.head(max_books)

    if display_df.empty:
        st.write("No books to show.")
        return

    for start in range(0, len(display_df), 5):
        row_slice = display_df.iloc[start:start+5]
        cols = st.columns(len(row_slice))

        for col, (_, book) in zip(cols, row_slice.iterrows()):
            with col:
                cover_id = book.get("cover_id", None)
                if pd.notna(cover_id):
                    cover_url = f"https://covers.openlibrary.org/b/id/{int(cover_id)}-M.jpg"
                    st.image(cover_url, use_container_width=False, width=110)

                title = str(book.get("title", ""))
                author = str(book.get("authors", ""))[:60]
                year = book.get("first_publish_year", "")

                st.markdown(f"**{title[:60]}{'…' if len(title) > 60 else ''}**")
                if author:
                    st.caption(author)
                if pd.notna(year):
                    st.caption(f"{int(year)}")


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

    # basic cols
    for col in ["title", "authors", "first_publish_year"]:
        if col not in df.columns:
            df[col] = ""

    # sidebar
    with st.sidebar:
        st.header("🔍 Filters")

        series_mode = st.radio(
            "Search only:",
            options=[
                "All books",
                "Books not in a series",
                "Books in a series",
            ],
            index=0,
        )

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

        page_buckets = st.multiselect("Page count", options=["<300", "300–499", "500+"])

        series_length_filters = st.multiselect(
            "Number of books in series",
            options=[
                "Standalone (1)",
                "Duology (2)",
                "Trilogy (3)",
                "Short series (4–6)",
                "Long series (7+)",
            ],
            help="Uses `series_length` column if available; otherwise this does nothing.",
        )

        st.markdown("---")

        # include/exclude pairs side by side
        st.markdown("**Mood**")
        c1, c2 = st.columns(2)
        with c1:
            include_moods = st.multiselect("Include", options=MOOD_OPTIONS, key="inc_mood")
        with c2:
            exclude_moods = st.multiselect("Exclude", options=MOOD_OPTIONS, key="exc_mood")

        st.markdown("**Pace**")
        c1, c2 = st.columns(2)
        with c1:
            include_pace = st.multiselect("Include", options=PACE_OPTIONS, key="inc_pace")
        with c2:
            exclude_pace = st.multiselect("Exclude", options=PACE_OPTIONS, key="exc_pace")

        st.markdown("**Type**")
        c1, c2 = st.columns(2)
        with c1:
            include_type = st.multiselect("Include", options=TYPE_OPTIONS, key="inc_type")
        with c2:
            exclude_type = st.multiselect("Exclude", options=TYPE_OPTIONS, key="exc_type")

        st.markdown("**Genre**")
        c1, c2 = st.columns(2)
        with c1:
            include_genre = st.multiselect("Include", options=GENRE_OPTIONS, key="inc_genre")
        with c2:
            exclude_genre = st.multiselect("Exclude", options=GENRE_OPTIONS, key="exc_genre")

        st.markdown("**Tropes**")
        c1, c2 = st.columns(2)
        with c1:
            include_tropes = st.multiselect("Include", options=TROPE_OPTIONS, key="inc_tropes")
        with c2:
            exclude_tropes = st.multiselect("Exclude", options=TROPE_OPTIONS, key="exc_tropes")

        st.markdown("**Hero / Heroine**")
        c1, c2 = st.columns(2)
        with c1:
            include_heroes = st.multiselect("Include", options=HERO_OPTIONS, key="inc_heroes")
        with c2:
            exclude_heroes = st.multiselect("Exclude", options=HERO_OPTIONS, key="exc_heroes")

        st.markdown("**Literary devices**")
        c1, c2 = st.columns(2)
        with c1:
            include_devices = st.multiselect("Include", options=DEVICE_OPTIONS, key="inc_devices")
        with c2:
            exclude_devices = st.multiselect("Exclude", options=DEVICE_OPTIONS, key="exc_devices")

        st.markdown("---")
        search = st.text_input(
            "Free text search (optional)",
            placeholder="title, author, trope, etc.",
        )

    # apply filters
    filtered, debug_messages = apply_filters(
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

    # debug info so you can see what’s happening
    with st.expander("Debug: filter steps", expanded=False):
        for msg in debug_messages:
            st.write("•", msg)

    st.subheader(f"Results ({len(filtered)})")

    if filtered.empty:
        st.warning("No books match your filters yet. Try relaxing some filters.")
        return

    st.markdown("### Books")
    render_book_grid(filtered)

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

        for label, col in [
            ("Type", "type"),
            ("Genre", "genre"),
            ("Mood", "mood"),
            ("Pace", "pace"),
            ("Tropes", "tropes"),
            ("Hero / Heroine", "hero_heroine"),
            ("Literary devices", "devices"),
            ("Series name", "series_name"),
        ]:
            if col in book and pd.notna(book.get(col)) and str(book.get(col)).strip():
                st.markdown(f"**{label}:** {book.get(col)}")

        if "page_count" in book and pd.notna(book.get("page_count")):
            st.markdown(f"**Page count:** {int(book['page_count'])}")

        if "series_length" in book and pd.notna(book.get("series_length")):
            st.markdown(f"**Series length:** {int(book['series_length'])} book(s)")


if __name__ == "__main__":
    main()
