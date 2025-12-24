from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pandas as pd


@dataclass(frozen=True)
class Query:
    champ: str | None = None
    top_killer_champ: str | None = None
    tag: str | None = None  # e.g., "multi-kill" or "objective-fight"
    min_kills: int | None = None
    min_participants: int | None = None
    match_id: str | None = None
    top_n_per_match: int | None = None
    sort_by: str = "kills_in_fight"
    descending: bool = True


def load_summaries(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # normalize necessary columns
    for c in ["champs_involved", "tags", "top_killer_champ"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)

    return df


def run_query(df: pd.DataFrame, q: Query) -> pd.DataFrame:
    out: pd.DataFrame = df.copy()

    if q.match_id:
        out = cast(pd.DataFrame, out[out["match_id"].astype(str) == q.match_id])

    if q.champ:
        champ = q.champ.strip().lower()
        out = cast(
            pd.DataFrame,
            out[
                out["champs_involved"]
                .str.lower()
                .str.contains(rf"(^|;){champ}(;|$)", regex=True)
            ],
        )

    if q.top_killer_champ:
        tk = q.top_killer_champ.strip().lower()
        out = cast(pd.DataFrame, out[out["top_killer_champ"].str.lower() == tk])

    if q.tag:
        tag = q.tag.strip().lower()
        out = cast(
            pd.DataFrame,
            out[out["tags"].str.lower().str.contains(rf"(^|;){tag}(;|$)", regex=True)],
        )

    if q.min_kills is not None and "kills_in_fight" in out.columns:
        out = cast(pd.DataFrame, out[out["kills_in_fight"] >= q.min_kills])

    if q.min_participants is not None and "participants_est" in out.columns:
        out = cast(pd.DataFrame, out[out["participants_est"] >= q.min_participants])

    sort_col = q.sort_by if q.sort_by in out.columns else None
    if sort_col is None:
        for candidate in ["kills_in_fight", "top_killer_kills", "participants_est"]:
            if candidate in out.columns:
                sort_col = candidate
                break

    if sort_col is not None:
        out = out.sort_values(sort_col, ascending=not q.descending)

    if q.top_n_per_match is not None and sort_col is not None:
        out = (
            out.sort_values(["match_id", sort_col], ascending=[True, not q.descending])
            .groupby("match_id", dropna=True)
            .head(q.top_n_per_match)
            .reset_index(drop=True)
        )
    else:
        out = out.reset_index(drop=True)

    return out


def save_query(df: pd.DataFrame, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path
