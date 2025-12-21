from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd

OBJECTIVE_TYPES = {"DRAGON", "BARON_NASHOR", "RIFTHERALD", "ATAKHAN"}


@dataclass
class TimelineTables:
    kills: pd.DataFrame
    objectives: pd.DataFrame


def parse_timeline(timeline_json: Dict[str, Any]) -> TimelineTables:
    """
    Parses Match-V5 timeline JSON into tidy tables.
    Focus: champion kills + elite monster objectives (dragon/baron/herald).
    """
    frames = timeline_json.get("info", {}).get("frames", [])
    kill_rows: List[Dict[str, Any]] = []
    obj_rows: List[Dict[str, Any]] = []

    for frame in frames:
        events = frame.get("events", [])
        for ev in events:
            etype = ev.get("type")
            t_ms = ev.get("timestamp", None)

            if etype == "CHAMPION_KILL":
                victims = ev.get("victimId")
                killer = ev.get("killerId")
                assists = ev.get("assistingParticipantIds", [])

                kill_rows.append(
                    {
                        "timestamp_ms": t_ms,
                        "timestamp_s": (t_ms / 1000.0) if t_ms is not None else None,
                        "killerId": killer,
                        "victimId": victims,
                        "assists": assists,
                        "position_x": ev.get("position", {}).get("x"),
                        "position_y": ev.get("position", {}).get("y"),
                        "bounty": ev.get("bounty"),
                        "shutdownBounty": ev.get("shutdownBounty"),
                    }
                )

            elif etype == "ELITE_MONSTER_KILL":
                mtype = ev.get("monsterType")
                if mtype in OBJECTIVE_TYPES:
                    obj_rows.append(
                        {
                            "timestamp_ms": t_ms,
                            "timestamp_s": (
                                (t_ms / 1000.0) if t_ms is not None else None
                            ),
                            "killerId": ev.get("killerId"),
                            "monsterType": mtype,
                            "monsterSubType": ev.get("monsterSubType"),
                            "teamId": ev.get("teamId"),
                            "position_x": ev.get("position", {}).get("x"),
                            "position_y": ev.get("position", {}).get("y"),
                        }
                    )

    df_kills = pd.DataFrame(kill_rows)
    df_obj = pd.DataFrame(obj_rows)

    # groups events into 30 second buckets
    if not df_kills.empty:
        df_kills["window_30s"] = (df_kills["timestamp_s"] // 30).astype("Int64")
        df_kills["n_assists"] = df_kills["assists"].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )
    if not df_obj.empty:
        df_obj["window_30s"] = (df_obj["timestamp_s"] // 30).astype("Int64")

    return TimelineTables(kills=df_kills, objectives=df_obj)
