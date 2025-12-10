import streamlit as st
import pandas as pd

from fablefinder import load_latest_catalogue, get_latest_catalogue_path


# ---------- helpers ----------

def explode_tokens(series: pd.Series, delimiters=(",",";")):
    """Turn a column of strings into a flat sorted list of unique tokens."""
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
    """Returns True if any candidate appears as substring in the cell."""
    if not candidates:
        return False
    if pd.isna(cell):
        return False
    text = str(cell).lower()
    return any(c.lower() in text for c in candidates)


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
) -> pd.DataFrame:
    filtered = df.copy()
    debug_steps = []

    debug_steps.append(f"Start: {len(filtered)} books")

    # --- year range ---
    if "first_publish_year" in filtered.columns:
        fp = filtered["first_publish_year"].fillna(0)
        before = len(filtered)
        filtered = filtered[(fp >= year_min) & (fp <= year_max)]
        debug_steps.append(f"After year filter: {len(filtered)} (was {before})")

    # --- series mode ---
    if series_mode == "Books not in a series":
        if "in_series" in filtered.columns:
            before = len(filtered)
            filtered = filtered[(filtered["in_series"] == False) | (filtered["in_series"].isna())]
            debug_steps.append(f"After 'not in series': {len(filtered)} (was {before})")
        elif "series_name" in filtered.columns:
            before = len(filtered)
            filtered = filtered[
                filtered["series_name"].isna()
                | (filtered["series_name"].astype(str).str.strip() == "")
            ]
            debug_steps.append(f"After 'not in series' via series_name: {len(filtered)} (was {before})")
        else:
            debug_steps.append("Series mode 'not in series' selected but no series column found.")
    elif series_mode == "Books in a series":
        if "in_series" in filtered.columns:
            before = len(filtered)
            filtered = filtered[filtered["in_series"] == True]
            debug_steps.append(f"After 'in series': {len(filtered)} (was {before})")
        elif "series_name" in filtered.columns:
            before = len(filtered)
            filtered = filtered[
                filtered["series_name"].notna()
                & (filtered["series_name"].astype(str).str.strip() != "")
            ]
            debug_steps.append(f"After 'in series' via series_name: {len(filtered)} (was {before})")
        else:
            debug_steps.append("Series mode 'in series' selected but no series column found.")

    # --- series length ---
    if "series_length" in filtered.columns and series_length_filters:
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
        debug_steps.append(f"After series length filter: {len(filtered)} (was {before})")
    elif series_length_filters:
        debug_steps.append("Series length filters selected but no 'series_length' column found.")

    # --- page count ---
    if "page_count" in filtered.columns and page_buckets:
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
        debug_steps.append(f"After page count filter: {len(filtered)} (was {before})")

    # helper
    def include_exclude(column_name: str, include_vals, exclude_vals, label: str):
        nonlocal filtered
        if column_name not in filtered.columns:
            if include_vals or exclude_vals:
                debug_steps.append(f"{label}: column '{column_name}' missing; ignoring include/exclude.")
            return

        col = filtered[column_name].astype(str)

        # include
        if include_vals:
            before = len(filtered)
            inc_mask = col.apply(lambda x: value_contains_any(x, include_vals))
            filtered = filtered[inc_mask]
            debug_steps.append(f"{label}: include {include_vals} -> {len(filtered)} (was {before})")

        # exclude
        if exclude_vals:
            before = len(filtered)
            exc_mask = col.apply(lambda x: value_contains_any(x, exclude_vals))
            filtered = filtered[~exc_mask]
            debug_steps.append(f"{label}: exclude {exclude_vals} -> {len(filtered)} (was {before})")

    include_exclude("mood", include_moods, exclude_moods, "Mood")
    include_exclude("pace", include_pace, exclude_pace, "Pace")
    include_exclude("type", include_type, exclude_type, "Type")
    include_exclude("genre", include_genre, exclude_genre, "Genre")
    include_exclude("tropes", include_tropes, exclude_tropes, "Tropes")
    include_exclude("hero_heroine", include_heroes, exclude_heroes, "Hero/Heroine")
    include_exclude("devices", include_devices, exclude_devices, "Devices")

    # --- free text search (optional) ---
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
            | col_contains("hero_heroine")
            | col_contains("devices")
            | col_contains("subjects")
        )
        filtered = filtered[mask]
        debug_steps.append(f"After search '{search}': {len(filtered)} (was {before})")

    return filtered, debug_steps


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
    st.caption("Filter your fantasy catalogue by mood, pace, series, tropes, and more.")

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

    for col in ["title", "authors", "first_publish_year"]:
        if col not in df.columns:
            df[col] = ""

    # build options from data only
    mood_options = explode_tokens(df["mood"]) if "mood" in df.columns else []
    pace_options = explode_tokens(df["pace"]) if "pace" in df.columns else []
    type_options = explode_tokens(df["type"]) if "type" in df.columns else []
    genre_options = explode_tokens(df["genre"]) if "genre" in df.columns else []
    trope_options = explode_tokens(df["tropes"]) if "tropes" in df.columns else []
    hero_options = explode_tokens(df["hero_heroine"]) if "hero_heroine" in df.columns else []
    device_options = explode_tokens(df["devices"]) if "devices" in df.columns else []

    # Sidebar
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

        bucket_options = ["<300", "300–499", "500+"]
        page_buckets = st.multiselect("Page count", options=bucket_options)

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

        # mood
        if mood_options:
            st.markdown("**Mood**")
            c1, c2 = st.columns(2)
            with c1:
                include_moods = st.multiselect("Include", options=mood_options, key="inc_mood")
            with c2:
                exclude_moods = st.multiselect("Exclude", options=mood_options, key="exc_mood")
        else:
            include_moods, exclude_moods = [], []

        # pace
        if pace_options:
            st.markdown("**Pace**")
            c1, c2 = st.columns(2)
            with c1:
                include_pace = st.multiselect("Include", options=pace_options, key="inc_pace")
            with c2:
                exclude_pace = st.multiselect("Exclude", options=pace_options, key="exc_pace")
        else:
            include_pace, exclude_pace = [], []

        # type
        if type_options:
            st.markdown("**Type**")
            c1, c2 = st.columns(2)
            with c1:
                include_type = st.multiselect("Include", options=type_options, key="inc_type")
            with c2:
                exclude_type = st.multiselect("Exclude", options=type_options, key="exc_type")
        else:
            include_type, exclude_type = [], []

        # genre
        if genre_options:
            st.markdown("**Genre**")
            c1, c2 = st.columns(2)
            with c1:
                include_genre = st.multiselect("Include", options=genre_options, key="inc_genre")
            with c2:
                exclude_genre = st.multiselect("Exclude", options=genre_options, key="exc_genre")
        else:
            include_genre, exclude_genre = [], []

        # tropes
        if trope_options:
            st.markdown("**Tropes**")
            c1, c2 = st.columns(2)
            with c1:
                include_tropes = st.multiselect("Include", options=trope_options, key="inc_tropes")
            with c2:
                exclude_tropes = st.multiselect("Exclude", options=trope_options, key="exc_tropes")
        else:
            include_tropes, exclude_tropes = [], []

        # hero/heroine
        if hero_options:
            st.markdown("**Hero / Heroine**")
            c1, c2 = st.columns(2)
            with c1:
                include_heroes = st.multiselect("Include", options=hero_options, key="inc_heroes")
            with c2:
                exclude_heroes = st.multiselect("Exclude", options=hero_options, key="exc_heroes")
        else:
            include_heroes, exclude_heroes = [], []

        # devices
        if device_options:
            st.markdown("**Literary devices**")
            c1, c2 = st.columns(2)
            with c1:
                include_devices = st.multiselect("Include", options=device_options, key="inc_devices")
            with c2:
                exclude_devices = st.multiselect("Exclude", options=device_options, key="exc_devices")
        else:
            include_devices, exclude_devices = [], []

        st.markdown("---")
        search = st.text_input(
            "Free text search (optional)",
            placeholder="title, author, trope, device, etc.",
        )

    # apply filters
    filtered, debug_steps = apply_filters(
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

    # debug summary
    with st.expander("Debug: filter steps", expanded=False):
        for msg in debug_steps:
            st.write("•", msg)

    st.subheader(f"Results ({len(filtered)})")

    if filtered.empty:
        st.warning("No books match your filters yet. Try relaxing one or two constraints.")
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
