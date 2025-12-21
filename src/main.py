from __future__ import annotations
from pathlib import Path
import json
import pandas as pd

from .services.timeline_parser import parse_timeline
from .services.data_fetcher import fetch_data


CACHE_DIR = Path("data/cache")
MATCH_DIR = CACHE_DIR / "matches"
TL_DIR = CACHE_DIR / "timelines"

OUT_DIR = Path("data/derived")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_window_features(df_kills: pd.DataFrame, df_obj: pd.DataFrame) -> pd.DataFrame:
    # Base index of windows from whichever df is non-empty
    windows = pd.Series(dtype="Int64")
    if not df_kills.empty:
        windows = df_kills["window_30s"].dropna()
    if windows.empty and not df_obj.empty:
        windows = df_obj["window_30s"].dropna()

    if windows.empty:
        cols = [
            "window_30s",
            "t_start_s",
            "t_end_s",
            "kill_count",
            "unique_participants",
            "unique_killers",
            "avg_assists",
            "objective_count",
            "dragon_count",
            "baron_count",
            "herald_count",
            "atakhan_count",
        ]
        return pd.DataFrame(columns=pd.Index(cols))

    win_index = pd.Index(sorted(windows.unique()), name="window_30s")
    feats = pd.DataFrame(index=win_index).reset_index()
    feats["t_start_s"] = feats["window_30s"] * 30
    feats["t_end_s"] = feats["t_start_s"] + 30

    if df_kills.empty:
        feats["kill_count"] = 0
        feats["unique_participants"] = 0
        feats["unique_killers"] = 0
        feats["avg_assists"] = 0.0
    else:
        grouped_kills = df_kills.groupby(
            "window_30s", dropna=True
        )  # each kill corresponds to a row. number of rows = number of kills in window

        feats = feats.merge(
            grouped_kills.size().to_frame("kill_count").reset_index(),
            on="window_30s",
            how="left",
        )

        # participants = killer + victim + assists
        def participants_in_window(sub: pd.DataFrame) -> int:
            ids = set()
            for _, row in sub.iterrows():
                killer = row.get("killerId")
                if killer is not None and not pd.isna(killer):
                    k = int(killer)
                    if k != 0:  # 0 is not a valid participant id from what i can tell
                        ids.add(int(killer))

                victim = row.get("victimId")
                if victim is not None and not pd.isna(victim):
                    v = int(victim)
                    if v != 0:  # 0 is not a valid participant id from what i can tell
                        ids.add(int(victim))

                assists = row.get("assists", [])
                if isinstance(assists, list):
                    for a in assists:
                        if a is not None and not pd.isna(a) and int(a) != 0:
                            ids.update([int(a) for a in assists])
            return len(ids)

        feats_part = (
            grouped_kills.apply(participants_in_window)
            .to_frame("unique_participants")
            .reset_index()
        )
        feats = feats.merge(feats_part, on="window_30s", how="left")

        feats = feats.merge(
            grouped_kills["killerId"]
            .nunique(dropna=True)
            .to_frame("unique_killers")
            .reset_index(),
            on="window_30s",
            how="left",
        )
        feats = feats.merge(
            grouped_kills["n_assists"].mean().to_frame("avg_assists").reset_index(),
            on="window_30s",
            how="left",
        )

    if df_obj.empty:
        feats["objective_count"] = 0
        feats["dragon_count"] = 0
        feats["baron_count"] = 0
        feats["herald_count"] = 0
        feats["atakhan_count"] = 0
    else:
        og = df_obj.groupby("window_30s", dropna=True)

        feats = feats.merge(
            og.size().to_frame("objective_count").reset_index(),
            on="window_30s",
            how="left",
        )

        # counts by type
        pivot = df_obj.pivot_table(
            index="window_30s",
            columns="monsterType",
            values="timestamp_s",
            aggfunc="count",
            fill_value=0,
        ).reset_index()

        for col, outcol in [
            ("DRAGON", "dragon_count"),
            ("BARON_NASHOR", "baron_count"),
            ("RIFTHERALD", "herald_count"),
            ("ATAKHAN", "atakhan_count"),
        ]:
            if col not in pivot.columns:
                pivot[col] = 0
            pivot = pivot.rename(columns={col: outcol})

        feats = feats.merge(
            pivot[
                [
                    "window_30s",
                    "dragon_count",
                    "baron_count",
                    "herald_count",
                    "atakhan_count",
                ]
            ],
            on="window_30s",
            how="left",
        )

    numeric_cols = [c for c in feats.columns if c != "window_30s"]
    feats[numeric_cols] = feats[numeric_cols].fillna(0)

    # ensure no columns are being inferred as objects for DBSCAN
    for col in [
        "kill_count",
        "unique_participants",
        "unique_killers",
        "objective_count",
        "dragon_count",
        "baron_count",
        "herald_count",
        "atakhan_count",
    ]:
        if col in feats.columns:
            feats[col] = feats[col].astype("int64")

    if "avg_assists" in feats.columns:
        feats["avg_assists"] = feats["avg_assists"].astype("float64")

    return feats


def main(refetch_data: bool = False):
    if refetch_data:
        fetch_data()

    timeline_files = sorted(TL_DIR.glob("*.json"))
    if not timeline_files:
        raise RuntimeError("No cached timelines found.")

    all_features = []

    for tl_path in timeline_files:
        match_id = tl_path.stem
        match_path = MATCH_DIR / f"{match_id}.json"
        if not match_path.exists():
            print(f"Skipping {match_id}: match details missing")
            continue

        tl_json = load_json(tl_path)
        tables = parse_timeline(tl_json)

        # save tidy event tables for inspection
        match_out_dir = OUT_DIR / match_id
        match_out_dir.mkdir(parents=True, exist_ok=True)

        if not tables.kills.empty:
            tables.kills.to_csv(match_out_dir / "kills.csv", index=False)
        else:
            (match_out_dir / "kills.csv").write_text("", encoding="utf-8")

        if not tables.objectives.empty:
            tables.objectives.to_csv(match_out_dir / "objectives.csv", index=False)
        else:
            (match_out_dir / "objectives.csv").write_text("", encoding="utf-8")

        feats = build_window_features(tables.kills, tables.objectives)
        feats["match_id"] = match_id
        feats.to_csv(match_out_dir / "window_features_30s.csv", index=False)

        all_features.append(feats)

        print(
            f"Processed {match_id}: windows={len(feats)} kills={len(tables.kills)} objs={len(tables.objectives)}"
        )

    if all_features:
        big = pd.concat(all_features, ignore_index=True)
        big.to_csv(OUT_DIR / "all_window_features_30s.csv", index=False)
        print(f"Saved combined features: {OUT_DIR / 'all_window_features_30s.csv'}")


if __name__ == "__main__":
    main()
