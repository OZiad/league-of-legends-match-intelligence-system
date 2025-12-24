# League of Legends Match Intelligence System

A data pipeline that turns raw League of Legends match timelines into **detected teamfights**, **fight summaries**, and **queryable highlights**.

This project answers questions like:
- “What were Jinx's top 3 objective fights?”
- “Show me Shaco multi-kill fights.”

---

## What This Does

1. **Fetches match timeline data** (kills, assists, objectives)
2. **Builds rolling 30s windows** of fight-related activity
3. **Clusters windows using DBSCAN** to detect teamfights
4. **Converts clusters into short fight segments** (≤ ~60s)
5. **Summarizes fights** (kills, champs, objectives, kill feed)
6. **Supports structured + natural language queries**

---

## Pipeline Overview

```
Riot API
   ↓
Windowed Features (30s)
   ↓
DBSCAN Clustering
   ↓
Fight Segments
   ↓
Fight Summaries
   ↓
Query Engine (NL or structured)
```

---

## Key Concepts

- **Windowed features**: Game events aggregated into fixed 30s slices
- **Fight detection**: Dense clusters of activity (kills + participants)
- **DBSCAN**: Density-based clustering (no fixed number of fights)
- **Fight summary**: Human-readable breakdown of a fight
- **Query module**: Filter, rank, and search fights

---

## Project Structure

```
src/
├── main.py                     # Pipeline entry point
├── services/
│   ├── datafetcher.py           # Riot API data fetch
│   ├── feature_extractor.py     # 30s window feature builder
│   ├── fight_detector.py        # DBSCAN fight detection
│   ├── fight_summarizer.py      # Fight summaries + kill feeds
│   ├── fight_query.py           # Structured queries
│   └── nl_query.py              # Natural language → Query
data/
├── cache/                         # Raw timeline data extracted from riot api
└── derived/
    ├── all_window_features_30s.csv
    ├── detected_fights.csv
    ├── fight_summaries.csv
    └── query_results.csv
```

---

## How Fights Are Detected

A **teamfight** is defined as:
- Multiple kills
- Many unique participants
- Occurring close together in time

Process:
1. Build numeric features per 30s window
2. Normalize features
3. Run **DBSCAN**
4. Split long clusters into short fight segments (≤ ~60s)

No hard-coded kill count rules.

---

## Running the Project

Install dependencies (using `uv`):

```bash
uv venv
uv sync
```

Run the full pipeline:

```bash
uv run python -m src.main
```

Outputs are written to:

```
data/derived/
```

---

## Querying Fights

### Example: Natural Language

```text
show me shaco multi-kill fights top 3 per match
```

### Example: Structured Query

```python
Query(
    champ="shaco",
    tag="multi-kill",
    top_n_per_match=3
)
```

Results are saved to:

```
data/derived/query_results.csv
```

---

## Why This Exists

Raw LoL timelines are noisy.

This system:
- Removes junk events
- Identifies **meaningful moments**
- Produces data usable for analytics, or ML
