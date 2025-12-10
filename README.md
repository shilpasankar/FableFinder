# FableFinder 🧙📚

FableFinder is a lightweight **fantasy book discovery app** built with [Streamlit](https://streamlit.io/).  
It helps readers find their next read based on **mood, subgenre, trope, and vibes**, not just ratings.

---

## ✨ Features

- Filter books by:
  - **Subgenre** (Epic, Urban, Cozy, Grimdark, YA, etc.)
  - **Mood/Vibe** (wholesome, dark, political, whimsical, romantic)
  - **Tropes** (found family, chosen one, heist, dragons, court politics, etc.)
- **Free-text search** over title, author, and tags.
- Simple “**You might also like**” recs based on:
  - Shared subgenre
  - Overlapping tropes/tags
  - Similar mood

Out of the box, FableFinder uses a **small demo dataset** defined in the app itself.  
You can also plug in your own CSV.

---

## 🏗 Tech Stack

- **Python 3.9+**
- **Streamlit**
- **Pandas**
- (Optional) **scikit-learn** for nicer similarity scoring

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/FableFinder.git
cd FableFinder
