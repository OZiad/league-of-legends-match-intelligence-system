from __future__ import annotations

from pathlib import Path

from src.services.data_fetcher import fetch_data
from src.services.feature_extractor import (
    FeatureExtractorConfig,
    extract_features_from_cache,
)
from src.services.fight_query import load_summaries, run_query, save_query
from src.services.fight_summarizer import summarize_fights
from src.services.nl_query import parse_nl
from src.services.teamfight_detector import DBSCANConfig, detect_teamfights


def main(refetch_data: bool = False):
    if refetch_data:
        fetch_data()

    features_csv = extract_features_from_cache(FeatureExtractorConfig())

    fights_csv = Path("data/derived/detected_fights.csv")
    detect_teamfights(
        features_csv=Path(features_csv),
        out_csv=fights_csv,
        dbscan_cfg=DBSCANConfig(eps=0.9, min_samples=2),
    )

    summaries_csv = Path("data/derived/fight_summaries.csv")
    summarize_fights(fights_csv, out_csv=summaries_csv)

    df = load_summaries(summaries_csv)

    allowed = set(
        df["champs_involved"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.split(";")
        .explode()
        .str.strip()
    )
    allowed.discard("")

    text = "show me shaco multikill fights top 3 per match"
    parsed = parse_nl(text, allowed_champs=allowed)

    for w in parsed.warnings:
        print(f"[query warning] {w}")

    q = parsed.query
    print("[parsed query]", q)

    result = run_query(df, q)

    out_path = save_query(result, Path("data/derived/query_results.csv"))
    print(f"Saved query results: {out_path} (rows={len(result)})")

    if result.empty:
        print("No results found.")
    else:
        preview_cols = [
            "match_id",
            "clip_start_s",
            "clip_end_s",
            "kills_in_fight",
            "participants_est",
            "top_killer_champ",
            "tags",
        ]
        preview_cols = [c for c in preview_cols if c in result.columns]
        print(result[preview_cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main(refetch_data=False)
