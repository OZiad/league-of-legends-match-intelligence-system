from __future__ import annotations

from pathlib import Path

from src.services.data_fetcher import fetch_data
from src.services.feature_extractor import (
    FeatureExtractorConfig,
    extract_features_from_cache,
)
from src.services.fight_summarizer import summarize_fights
from src.services.teamfight_detector import DBSCANConfig, detect_teamfights


def main(refetch_data: bool = False):
    if refetch_data:
        fetch_data()

    features_csv = extract_features_from_cache(FeatureExtractorConfig())

    detect_teamfights(
        features_csv=Path(features_csv),
        out_csv=Path("data/derived/detected_fights.csv"),
        dbscan_cfg=DBSCANConfig(eps=0.9, min_samples=2),
    )
    fights_csv = Path("data/derived/detected_fights.csv")
    summarize_fights(fights_csv, out_csv=Path("data/derived/fight_summaries.csv"))


if __name__ == "__main__":
    main(refetch_data=False)
