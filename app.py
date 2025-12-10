import streamlit as st
import pandas as pd

from fablefinder import load_latest_catalogue, get_latest_catalogue_path


def apply_filters(df: pd.DataFrame, search: str, year_min: int, year_max: int) -> pd.DataFrame:
    filtered = df.copy()

    # Year filter
    if "first_publish_year" in filtered.columns:
        filtered = filtered[
            (filtered["first_publish_year"].fillna(0) >= year_min)
            & (filtered["first_publish_year"].fillna(9999) <= year_max)
        ]

    # Text search
    if search:
        s = search.lower()
        mask = (
            filtered.get("title", "").astype(str).str.lower().str.contains(s)
            | filtered.get("authors", "").astype(str).str.lower().str.contains(s)
            | filtered.get("subjects", "").astype(str).str.lower().str.contains(s)
        )
        filtered = filtered[mask]

    return filtered


def main():
    st.set_page_config(
        page_title="FableFinder Catalogue",
        page_icon="📚",
        layout="wide",
    )

    st.title("FableFinder 📚")
    st.caption("Fantasy catalogue powered by Open Library, updated weekly.")

    # Info about which file we’re using
    latest_path = get_latest_catalogue_path()
    if latest_path is None:
        st.error(
            "No catalogue files found in `data/`.\n\n"
            "Run `python fablefinder.py` to create `catalogue_YYMMDD.csv` first."
        )
        st.stop()

    st.info(f"Using latest catalogue: `{latest_path.name}`")

    # Load data
    try:
        df = load_latest_catalogue()
    except Exception as e:
        st.error(f"Could not load latest catalogue: {e}")
        st.stop()

    # Ensure some cols exist
    for col in ["title", "authors", "first_publish_year", "subjects"]:
        if col not in df.columns:
            df[col] = ""

    # Sidebar filters
    with st.sidebar:
        st.header("🔍 Filters")
        search = st.text_input("Search (title, author, subject)", "")

        # Year slider based on data
        min_year = int(df["first_publish_year"].min()) if df["first_publish_year"].notna().any() else 1800
        max_year = int(df["first_publish_year"].max()) if df["first_publish_year"].notna().any() else 2025

        year_min, year_max = st.slider(
            "First publish year range",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year),
        )

        st.markdown("---")
        st.caption("Catalogue is generated via `fablefinder.py` (Open Library).")

    filtered = apply_filters(df, search, year_min, year_max)

    st.subheader(f"Catalogue results ({len(filtered)})")

    if filtered.empty:
        st.warning("No books match your filters yet. Try adjusting search/year range.")
        return

    # Main display: table + details
    left, right = st.columns([1.4, 1])

    with left:
        st.dataframe(
            filtered[["title", "authors", "first_publish_year", "subjects"]],
            use_container_width=True,
            hide_index=True,
        )

    with right:
        st.subheader("Book details")

        options = [
            f"{row['title']} — {row['authors']} ({int(row['first_publish_year']) if pd.notna(row['first_publish_year']) else 'n/a'})"
            for _, row in filtered.iterrows()
        ]
        selected = st.selectbox("Pick a book", options)

        # Map back to row
        selected_idx = options.index(selected)
        book = filtered.iloc[selected_idx]

        st.markdown(f"### {book['title']}")
        st.markdown(f"*by {book['authors']}*")

        if pd.notna(book.get("first_publish_year", None)):
            st.markdown(f"**First published:** {int(book['first_publish_year'])}")

        st.markdown("**Subjects / tags:**")
        st.write(book.get("subjects", ""))

        # If we have a cover_id, show cover
        cover_id = book.get("cover_id", None)
        if not pd.isna(cover_id):
            cover_url = f"https://covers.openlibrary.org/b/id/{int(cover_id)}-L.jpg"
            st.image(cover_url, caption="Open Library cover", use_container_width=True)


if __name__ == "__main__":
    main()
