from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Optional


class JsonCache:
    def __init__(self, root: str = "data/cache"):
        self.root = Path(root)
        self.matches_dir = self.root / "matches"
        self.timelines_dir = self.root / "timelines"
        self.matches_dir.mkdir(parents=True, exist_ok=True)
        self.timelines_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, kind: str, match_id: str) -> Path:
        if kind == "match":
            return self.matches_dir / f"{match_id}.json"
        if kind == "timeline":
            return self.timelines_dir / f"{match_id}.json"
        raise ValueError(f"Unknown kind: {kind}")

    def get(self, kind: str, match_id: str) -> Optional[dict[str, Any]]:
        p = self._path(kind, match_id)
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def set(self, kind: str, match_id: str, payload: dict[str, Any]) -> None:
        p = self._path(kind, match_id)
        p.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
