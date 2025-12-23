from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class DBSCANConfig:
    eps: float = 0.9
    min_samples: int = 2


@dataclass(frozen=True)
class FightScoringConfig:
    # simple scoring used to rank detected clusters.
    kill_weight: float = 3.0
    participants_weight: float = 1.5
    objective_weight: float = 4.0
    baron_bonus: float = 6.0
    dragon_bonus: float = 3.0
    herald_bonus: float = 2.0
    atakhan_bonus: float = 4.0


def _required_cols_exist(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")


def run_dbscan_on_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    cfg: DBSCANConfig = DBSCANConfig(),
) -> pd.DataFrame:
    """
    Returns a copy of df with a new 'cluster_id' column:
      -1 = noise (not part of a cluster)
      0..N = cluster labels
    """
    _required_cols_exist(df, ["match_id", "t_start_s", "t_end_s"] + feature_cols)

    # keep only rows that have meaningful signal (optional but helps)
    work = df.copy()
    for c in feature_cols:
        col = cast(pd.Series, pd.to_numeric(work[c], errors="coerce"))
        work[c] = col.fillna(0)

    X = work[feature_cols].to_numpy(dtype=float)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = DBSCAN(eps=cfg.eps, min_samples=cfg.min_samples)
    labels = model.fit_predict(Xs)

    work["cluster_id"] = labels.astype(int)
    return work


def score_window_row(row: pd.Series, s: FightScoringConfig) -> float:
    obj_score = s.objective_weight * float(row.get("objective_count") or 0)
    obj_score += s.baron_bonus * float(row.get("baron_count") or 0)
    obj_score += s.dragon_bonus * float(row.get("dragon_count") or 0)
    obj_score += s.herald_bonus * float(row.get("herald_count") or 0)
    obj_score += s.atakhan_bonus * float(row.get("atakhan_count") or 0)

    return (
        s.kill_weight * float(row.get("kill_count") or 0)
        + s.participants_weight * float(row.get("unique_participants") or 0)
        + obj_score
    )


def clusters_to_fights(
    df_with_clusters: pd.DataFrame,
    scoring: FightScoringConfig = FightScoringConfig(),
    window_seconds: int = 30,
    max_gap_windows: int = 1,
) -> pd.DataFrame:
    """
    Convert per-window cluster assignments into contiguous fight segments per match.

    we split each (match_id, cluster_id) group
    into contiguous segments whenever there is a time gap.

    A new segment starts when:
        curr.t_start_s - prev.t_end_s > window_seconds * max_gap_windows
    """
    df = df_with_clusters.copy()

    # ignore noise
    df = df[df["cluster_id"] >= 0].copy()
    if df.empty:
        return pd.DataFrame(
            columns=pd.Index(
                [
                    "match_id",
                    "cluster_id",
                    "segment_id",
                    "fight_start_s",
                    "fight_end_s",
                    "duration_s",
                    "window_count",
                    "kills_total",
                    "kills_peak",
                    "participants_peak",
                    "objectives_total",
                    "barons_total",
                    "dragons_total",
                    "heralds_total",
                    "atakhans_total",
                    "fight_score",
                ]
            )
        )

    df["window_score"] = df.apply(lambda r: score_window_row(r, scoring), axis=1)

    df = df.sort_values(by=["match_id", "cluster_id", "t_start_s"]).reset_index(
        drop=True
    )
    max_gap_s = window_seconds * max_gap_windows

    # time gap from previous row within same (match_id, cluster_id)
    prev_end = df.groupby(["match_id", "cluster_id"], dropna=True)["t_end_s"].shift(1)
    gap = df["t_start_s"] - prev_end

    # start a new segment if first row in group or gap is bigger than max gap
    new_segment = prev_end.isna() | (gap > max_gap_s)

    # cumulative sum of "new segment" flags gives segment numbering per group
    df["segment_id"] = (
        new_segment.groupby([df["match_id"], df["cluster_id"]]).cumsum().astype(int) - 1
    )

    grouped = df.groupby(["match_id", "cluster_id", "segment_id"], dropna=True)

    fights = grouped.agg(
        fight_start_s=("t_start_s", "min"),
        fight_end_s=("t_end_s", "max"),
        window_count=("t_start_s", "count"),
        kills_total=("kill_count", "sum"),
        kills_peak=("kill_count", "max"),
        participants_peak=("unique_participants", "max"),
        objectives_total=("objective_count", "sum"),
        barons_total=("baron_count", "sum"),
        dragons_total=("dragon_count", "sum"),
        heralds_total=("herald_count", "sum"),
        atakhans_total=("atakhan_count", "sum"),
        fight_score=("window_score", "sum"),
    ).reset_index()

    fights["duration_s"] = fights["fight_end_s"] - fights["fight_start_s"]

    # Sort within match by score desc
    fights = fights.sort_values(
        ["match_id", "fight_score"], ascending=[True, False]
    ).reset_index(drop=True)

    return fights


def detect_teamfights(
    features_csv: Path,
    out_csv: Path = Path("data/derived/detected_fights.csv"),
    dbscan_cfg: DBSCANConfig = DBSCANConfig(),
    scoring_cfg: FightScoringConfig = FightScoringConfig(),
) -> Path:
    df = pd.read_csv(features_csv)

    feature_cols = [
        "kill_count",
        "unique_participants",
        "avg_assists",
        "objective_count",
        "baron_count",
        "dragon_count",
        "herald_count",
        "atakhan_count",
    ]
    _required_cols_exist(df, ["match_id", "t_start_s", "t_end_s"] + feature_cols)

    clustered = run_dbscan_on_windows(df, feature_cols=feature_cols, cfg=dbscan_cfg)
    fights = clusters_to_fights(clustered, scoring=scoring_cfg)

    fights = fights[
        (fights["window_count"] >= 2) & (fights["participants_peak"] >= 6)
    ].copy()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fights.to_csv(out_csv, index=False)
    print(f"Saved fights: {out_csv} (count={len(fights)})")

    return out_csv
