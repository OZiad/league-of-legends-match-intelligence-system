from __future__ import annotations

from src.services.data_fetcher import fetch_data
from src.services.feature_extractor import (
    extract_features_from_cache,
    FeatureExtractorConfig,
)


def main(refetch_data: bool = False):
    if refetch_data:
        fetch_data()

    extract_features_from_cache(FeatureExtractorConfig())


if __name__ == "__main__":
    main(refetch_data=False)
