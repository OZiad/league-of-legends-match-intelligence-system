from __future__ import annotations

from pathlib import Path

from src.services.data_fetcher import fetch_data
from src.services.feature_extractor import (
    FeatureExtractorConfig,
    extract_features_from_cache,
)
from src.services.fight_summarizer import summarize_fights
from src.services.teamfight_detector import DBSCANConfig, detect_teamfights
from src.services.fight_query import load_summaries, run_query, save_query, Query


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

    q = Query(
        tag="multi-kill",
        min_participants=6,
        sort_by="kills_in_fight",
        descending=True,
    )

    result = run_query(df, q)
    out_path = save_query(result, Path("data/derived/query_results.csv"))

    cols = [
        "match_id",
        "fight_start_s",
        "fight_end_s",
        "clip_start_s",
        "clip_end_s",
        "kills_in_fight",
        "participants_est",
        "top_killer_champ",
        "top_killer_kills",
        "tags",
    ]
    print(result[[c for c in cols if c in result.columns]].head(20))
    print(f"Saved query results: {out_path}")


if __name__ == "__main__":
    main(refetch_data=False)
