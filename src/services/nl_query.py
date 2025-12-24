from __future__ import annotations

import re
from dataclasses import dataclass, replace


from .fight_query import Query

KNOWN_TAGS = {
    "multi-kill": ["multi", "multikill", "multi-kill"],
    "objective-fight": ["objective", "objectives", "objective-fight", "obj"],
}


# champ detection is implemented in main
@dataclass(frozen=True)
class ParseResult:
    query: Query
    warnings: list[str]


_int_pat = re.compile(r"\b(\d+)\b")


def _find_int_after(words: list[str], key: str) -> int | None:
    # e.g. ["top","3","per","match"] with key="top" -> 3
    for i, w in enumerate(words[:-1]):
        if w == key:
            m = _int_pat.fullmatch(words[i + 1])
            if m:
                return int(m.group(1))
    return None


def parse_nl(text: str, *, allowed_champs: set[str] | None = None) -> ParseResult:
    raw = text.strip()
    s = raw.lower()

    # normalize punctuation into spaces
    s = re.sub(r"[^a-z0-9\-\s]", " ", s)
    words = [w for w in s.split() if w]

    q = Query()
    warnings: list[str] = []

    tags_found: list[str] = []
    for tag, aliases in KNOWN_TAGS.items():
        for a in aliases:
            if a in s:
                tags_found.append(tag)
                break
    if tags_found:
        q = replace(q, tag=tags_found[0])

    # stuff like: "top killer missfortune" or "topkiller missfortune"
    if "top" in words and "killer" in words:
        try:
            i = words.index("killer")
            if i + 1 < len(words):
                cand = words[i + 1]
                if allowed_champs and cand.lower() not in allowed_champs:
                    warnings.append(f"Unknown champion for top killer: {cand}")
                else:
                    q = replace(q, top_killer_champ=cand)

        except ValueError:
            pass

    # ---- champs involved ----
    # for stuff like "show me shaco fights", "fights with shaco", "shaco multi kill" we'll scan all tokens and pick the first token that looks like a champ in the allowed list
    if allowed_champs:
        for w in words:
            if w in allowed_champs:
                q = replace(q, champ=w)

                break

    # ---- numeric constraints ----
    # meant for stuff like "at least 6 participants" / "min 6 participants"
    if "participants" in words or "participant" in words:
        for key in ("least", "min"):
            n = _find_int_after(words, key)
            if n is not None:
                q = replace(q, min_participants=n)
                break

    if "kills" in words or "kill" in words:
        for key in ("least", "min"):
            n = _find_int_after(words, key)
            if n is not None:
                q = replace(q, min_kills=n)
                break

    # ---- top N per match ----
    n_top = _find_int_after(words, "top")
    if n_top is not None and ("match" in words or "matches" in words):
        q = replace(q, top_n_per_match=n_top)

    # allow "sort by kills" or "sort by participants"
    if "sort" in words and "by" in words:
        try:
            i = words.index("by")
            if i + 1 < len(words):
                field = words[i + 1]
                mapping = {
                    "kills": "kills_in_fight",
                    "participants": "participants_est",
                    "score": "fight_score",
                }
                if field in mapping:
                    q = replace(q, sort_by=mapping[field])

                else:
                    warnings.append(f"Unknown sort field: {field}")
        except ValueError:
            pass

    return ParseResult(query=q, warnings=warnings)
