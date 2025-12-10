import streamlit as st
import pandas as pd

from fablefinder import load_latest_catalogue, get_latest_catalogue_path


# ---------- helpers ----------

def value_contains_any(cell: str, candidates) -> bool:
    """True if any candidate appears (substring) in the cell."""
    if not candidates:
        return False
    if pd.isna(cell):
        return False
    text = str(cell).lower()
    return any(c.lower() in text for c in candidates)


def render_book_grid(df: pd.DataFrame, max_books: int = 50):
    """Show books as small cards in a grid (5 per row)."""
    display_df = df.head(max_books)

    if display_df.empty:
        st.write("No books to show.")
        return

    for start in range(0, len(display_df), 5):
        row_slice = display_df.iloc[start : start + 5]
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


def apply_filters(
    df: pd.DataFrame,
    search: str,
    year_min: int,
    year_max: int,
    include_genres,
    exclude_genres,
    include_tags,
    exclude_tags,
    series_mode: str,
):
    debug = []

    filtered = df.copy()
    debug.append(f"Start: {len(filtered)} books")

    # year filter
    if "first_publish_year" in filtered.columns:
        fp = filtered["first_publish_year"].fillna(0)
        before = len(filtered)
        filtered = filtered[(fp >= year_min) & (fp <= year_max)]
        debug.append(f"Year {year_min}-{year_max}: {len(filtered)} (was {before})")
    else:
        debug.append("No 'first_publish_year' column – year filter skipped.")

    # series mode (only if columns exist)
    if series_mode != "All books":
        if "in_series" in filtered.columns:
            before = len(filtered)
            if series_mode == "Books not in a series":
                filtered = filtered[(filtered["in_series"] == False) | (filtered["in_series"].isna())]
            else:  # Books in a series
                filtered = filtered[filtered["in_series"] == True]
            debug.append(f"{series_mode}: {len(filtered)} (was {before})")
        elif "series_name" in filtered.columns:
            before = len(filtered)
            if series_mode == "Books not in a series":
                filtered = filtered[
                    filtered["series_name"].isna()
                    | (filtered["series_name"].astype(str).str.strip() == "")
                ]
            else:
                filtered = filtered[
                    filtered["series_name"].notna()
                    & (filtered["series_name"].astype(str).str.strip() != "")
                ]
            debug.append(f"{series_mode} via series_name: {len(filtered)} (was {before})")
        else:
            debug.append(f"{series_mode} selected but no series column found; ignoring.")

    # genre include/exclude
    if "genre" in filtered.columns:
        if include_genres:
            before = len(filtered)
            inc_mask = filtered["genre"].astype(str).apply(lambda x: value_contains_any(x, include_genres))
            filtered = filtered[inc_mask]
            debug.append(f"Genre include {include_genres}: {len(filtered)} (was {before})")
        if exclude_genres and not filtered.empty:
            before = len(filtered)
            exc_mask = filtered["genre"].astype(str).apply(lambda x: value_contains_any(x, exclude_genres))
            filtered = filtered[~exc_mask]
            debug.append(f"Genre exclude {exclude_genres}: {len(filtered)} (was {before})")
    else:
        if include_genres or exclude_genres:
            debug.append("Genre filters selected but no 'genre' column in data.")

    # subjects/tag include/exclude
    if "subjects" in filtered.columns:
        if include_tags:
            before = len(filtered)
            inc_mask = filtered["subjects"].astype(str).apply(lambda x: value_contains_any(x, include_tags))
            filtered = filtered[inc_mask]
            debug.append(f"Tags include {include_tags}: {len(filtered)} (was {before})")
        if exclude_tags and not filtered.empty:
            before = len(filtered)
            exc_mask = filtered["subjects"].astype(str).apply(lambda x: value_contains_any(x, exclude_tags))
            filtered = filtered[~exc_mask]
            debug.append(f"Tags exclude {exclude_tags}: {len(filtered)} (was {before})")
    else:
        if include_tags or exclude_tags:
            debug.append("Tag filters selected but no 'subjects' column in data.")

    # text search (optional)
    if search:
        s = search.lower()
        before = len(filtered)

        def col_contains(col):
            if col not in filtered.columns:
                return pd.Series(False, index=filtered.index)
            return filtered[col].astype(str).str.lower().str.contains(s)

        mask = (
            col_contains("title")
            | col_contains("authors")
            | col_contains("genre")
            | col_contains("subjects")
            | col_contains("summary")
        )
        filtered = filtered[mask]
        debug.append(f"Search '{search}': {len(filtered)} (was {before})")
    else:
        debug.append("No text search applied.")

    return filtered, debug


# ---------- main UI ----------

def main():
    st.set_page_config(
        page_title="FableFinder",
        page_icon="📚",
        layout="wide",
    )

    st.title("FableFinder 🧙📚")
    st.caption("Explore your fantasy catalogue using real metadata (year, genre, tags, summary).")

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

    # Basic columns
    for col in ["title", "authors", "first_publish_year"]:
        if col not in df.columns:
            df[col] = ""

    # build genre/tag options based on actual data
    genre_options = []
    if "genre" in df.columns:
        genre_options = sorted(
            {g.strip() for g in df["genre"].dropna().astype(str).str.split(";").explode() if g.strip()}
        )

    tag_options = []
    if "subjects" in df.columns:
        tag_options = sorted(
            {s.strip() for s in df["subjects"].dropna().astype(str).str.split(",").explode() if s.strip()}
        )

    # sidebar
    with st.sidebar:
        st.header("🔍 Filters")

        # series mode
        series_mode = st.radio(
            "Search only:",
            options=[
                "All books",
                "Books not in a series",
                "Books in a series",
            ],
            index=0,
        )

        # year range
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

        st.markdown("---")

        # genre include/exclude
        if genre_options:
            st.markdown("**Genre (from CMU dataset)**")
            c1, c2 = st.columns(2)
            with c1:
                include_genres = st.multiselect("Include", options=genre_options, key="inc_genre")
            with c2:
                exclude_genres = st.multiselect("Exclude", options=genre_options, key="exc_genre")
        else:
            include_genres = []
            exclude_genres = []
            st.caption("No 'genre' column detected – CMU enrichment may not be running?")

        # subjects / tags include/exclude
        if tag_options:
            st.markdown("**Subject tags (from Open Library)**")
            c1, c2 = st.columns(2)
            with c1:
                include_tags = st.multiselect("Include", options=tag_options, key="inc_tags")
            with c2:
                exclude_tags = st.multiselect("Exclude", options=tag_options, key="exc_tags")
        else:
            include_tags = []
            exclude_tags = []
            st.caption("No 'subjects' column detected – did Open Library fetch succeed?")

        st.markdown("---")
        search = st.text_input(
            "Free text search (optional)",
            placeholder="title, author, genre, subject, summary...",
        )

    # apply filters
    filtered, debug = apply_filters(
        df=df,
        search=search,
        year_min=year_min,
        year_max=year_max,
        include_genres=include_genres,
        exclude_genres=exclude_genres,
        include_tags=include_tags,
        exclude_tags=exclude_tags,
        series_mode=series_mode,
    )

    with st.expander("Debug: filter steps", expanded=False):
        for line in debug:
            st.write("•", line)

    st.subheader(f"Results ({len(filtered)})")

    if filtered.empty:
        st.warning("No books match your filters yet. Try relaxing one or two filters.")
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

        if "genre" in book and pd.notna(book.get("genre")):
            st.markdown(f"**Genre:** {book.get('genre')}")

        if "subjects" in book and pd.notna(book.get("subjects")):
            st.markdown(f"**Subject tags:** {book.get('subjects')}")

        if "summary" in book and pd.notna(book.get("summary")):
            with st.expander("Summary", expanded=True):
                st.write(book.get("summary"))


if __name__ == "__main__":
    main()
