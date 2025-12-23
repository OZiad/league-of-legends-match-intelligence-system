from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

CACHE_DIR = Path("data/cache")
MATCH_DIR = CACHE_DIR / "matches"
TL_DIR = CACHE_DIR / "timelines"


@dataclass(frozen=True)
class ClipWindowConfig:
    pre_s: int = 8  # time before fight start to include
    post_s: int = 6  # seconds after fight end to include


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _participant_champ_map(match_json: dict[str, Any]) -> dict[int, str]:
    out: dict[int, str] = {}
    info = match_json.get("info", {})
    parts = info.get("participants", [])
    for p in parts:
        pid = p.get("participantId")
        champ = p.get("championName") or p.get("championId") or "Unknown"
        if isinstance(pid, int):
            out[pid] = str(champ)
    return out


def _iter_events_in_range(
    tl_json: dict[str, Any], t0_s: int, t1_s: int
) -> list[dict[str, Any]]:
    info = tl_json.get("info", {})
    frames = info.get("frames", [])
    out: list[dict[str, Any]] = []

    t0_ms = t0_s * 1000
    t1_ms = t1_s * 1000

    for fr in frames:
        events = fr.get("events", [])
        for ev in events:
            ts = ev.get("timestamp")
            if not isinstance(ts, int):
                continue
            if t0_ms <= ts <= t1_ms:
                out.append(ev)
    return out


def _summarize_kills(
    events: list[dict[str, Any]], pid_to_champ: dict[int, str]
) -> tuple[list[str], dict[int, int], set[int]]:
    """
    Returns:
      - kill_feed lines
      - killer_counts (participantId -> kills)
      - involved_ids (set of participantIds seen in kills)
    """
    kill_feed: list[str] = []
    killer_counts: dict[int, int] = {}
    involved: set[int] = set()

    for ev in events:
        if ev.get("type") != "CHAMPION_KILL":
            continue

        killer = ev.get("killerId")
        victim = ev.get("victimId")
        assists = ev.get("assistingParticipantIds", []) or ev.get("assists", [])

        if isinstance(killer, int) and killer != 0:
            killer_counts[killer] = killer_counts.get(killer, 0) + 1
            involved.add(killer)
            k_name = pid_to_champ.get(killer, f"{killer}")
        else:
            k_name = "Unknown"

        if isinstance(victim, int) and victim != 0:
            involved.add(victim)
            v_name = pid_to_champ.get(victim, f"P{victim}")
        else:
            v_name = "Unknown"

        assist_ids: list[int] = []
        if isinstance(assists, list):
            for a in assists:
                if isinstance(a, int) and a != 0:
                    assist_ids.append(a)
                    involved.add(a)

        a_names = [pid_to_champ.get(a, f"P{a}") for a in assist_ids]

        if a_names:
            kill_feed.append(f"{k_name} -> {v_name} (assists: {', '.join(a_names)})")
        else:
            kill_feed.append(f"{k_name} -> {v_name}")

    return kill_feed, killer_counts, involved


def _summarize_objectives(events: list[dict[str, Any]]) -> dict[str, int]:
    out = {
        "dragon": 0,
        "baron": 0,
        "herald": 0,
        "atakhan": 0,
        "tower": 0,
        "inhib": 0,
    }

    for ev in events:
        t = ev.get("type")

        if t == "ELITE_MONSTER_KILL":
            m = ev.get("monsterType")
            if m == "DRAGON":
                out["dragon"] += 1
            elif m == "BARON_NASHOR":
                out["baron"] += 1
            elif m == "RIFTHERALD":
                out["herald"] += 1
            elif m == "ATAKHAN":
                out["atakhan"] += 1

        elif t == "BUILDING_KILL":
            b = ev.get("buildingType")
            if b == "TOWER_BUILDING":
                out["tower"] += 1
            elif b == "INHIBITOR_BUILDING":
                out["inhib"] += 1

    return out


def summarize_fights(
    fights_csv: Path,
    out_csv: Path = Path("data/derived/fight_summaries.csv"),
    clip_cfg: ClipWindowConfig = ClipWindowConfig(),
) -> Path:
    fights = pd.read_csv(fights_csv)

    rows_out: list[dict[str, Any]] = []

    for _, f in fights.iterrows():
        match_id = str(f["match_id"])
        fight_start_s = int(f["fight_start_s"])
        fight_end_s = int(f["fight_end_s"])

        clip_start_s = max(0, fight_start_s - clip_cfg.pre_s)
        clip_end_s = fight_end_s + clip_cfg.post_s

        match_path = MATCH_DIR / f"{match_id}.json"
        tl_path = TL_DIR / f"{match_id}.json"
        if not match_path.exists() or not tl_path.exists():
            continue

        match_json = _load_json(match_path)
        tl_json = _load_json(tl_path)

        pid_to_champ = _participant_champ_map(match_json)

        events = _iter_events_in_range(tl_json, fight_start_s, fight_end_s)
        kill_feed, killer_counts, involved_ids = _summarize_kills(events, pid_to_champ)
        obj = _summarize_objectives(events)

        # naive approach of assuming the "highlight player" is the person with the most kills in the fight
        top_killer = None
        top_kills = 0
        for pid, k in killer_counts.items():
            if k > top_kills:
                top_killer = pid
                top_kills = k

        champs_involved = sorted(
            {pid_to_champ.get(pid, f"P{pid}") for pid in involved_ids}
        )

        tags: list[str] = []
        if top_kills >= 3:
            tags.append("multi-kill")
        if obj["dragon"] + obj["baron"] + obj["herald"] + obj["atakhan"] > 0:
            tags.append("objective-fight")

        rows_out.append(
            {
                "match_id": match_id,
                "cluster_id": int(f.get("cluster_id") or -1),
                "segment_id": int(f.get("segment_id") or -1),
                "fight_start_s": fight_start_s,
                "fight_end_s": fight_end_s,
                "clip_start_s": clip_start_s,
                "clip_end_s": clip_end_s,
                "kills_in_fight": len(
                    [x for x in events if x.get("type") == "CHAMPION_KILL"]
                ),
                "participants_est": len(involved_ids),
                "top_killer_participantId": (
                    int(top_killer) if top_killer is not None else -1
                ),
                "top_killer_champ": (
                    pid_to_champ.get(int(top_killer), "Unknown")
                    if top_killer is not None
                    else "None"
                ),
                "top_killer_kills": int(top_kills),
                "obj_dragon": obj["dragon"],
                "obj_baron": obj["baron"],
                "obj_herald": obj["herald"],
                "obj_atakhan": obj["atakhan"],
                "obj_tower": obj["tower"],
                "obj_inhib": obj["inhib"],
                "champs_involved": ";".join(champs_involved),
                "kill_feed": " | ".join(kill_feed[:8]),  # cap to keep CSV readable
                "tags": ";".join(tags),
            }
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_out).to_csv(out_csv, index=False)
    print(f"Saved fight summaries: {out_csv} (count={len(rows_out)})")
    return out_csv
