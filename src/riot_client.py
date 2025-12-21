from __future__ import annotations
import os
from typing import Any, Optional
import requests
import time


class RiotClient:
    def __init__(self, api_key: Optional[str] = None, timeout_s: int = 30):
        key = api_key or os.getenv("RIOT_API_KEY")
        if not key:
            raise RuntimeError("API key not found")

        self.api_key: str = key
        self.timer = timeout_s

    def _headers(self) -> dict[str, str]:
        return {"X-Riot-Token": self.api_key}

    def _get(
        self, url: str, params: Optional[dict[str, str]] = None, max_retries: int = 5
    ):
        for attempt in range(max_retries):
            r = requests.get(
                url, headers=self._headers(), params=params, timeout=self.timer
            )

            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After")
                sleep_s = (
                    int(retry_after)
                    if retry_after and retry_after.isdigit()
                    else (2**attempt)
                )
                time.sleep(sleep_s)
                continue

            if 200 <= r.status_code < 300:
                return r.json()

            else:
                try:
                    detail = r.json()
                except Exception:
                    detail = {"text": r.text[:300]}

                raise RuntimeError(f"HTTP {r.status_code} for {url} | detail={detail}")

        raise RuntimeError(f"Exceeded retries for {url}")

    @staticmethod
    def _regional_host(regional_routing: str) -> str:
        return f"https://{regional_routing}.api.riotgames.com"

    def get_puuid_by_riot_id(
        self, game_name: str, tag_line: str, regional_routing: str
    ) -> str:
        base = self._regional_host(regional_routing)
        url = f"{base}/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}"
        data = self._get(url)
        return data["puuid"]

    """
    /lol/match/v5/matches/by-puuid/{puuid}/ids/lol/match/v5/matches/by-puuid/{puuid}/ids
    """

    def get_match_ids_by_puuid(
        self,
        puuid: str,
        regional_routing: str,
        start: int = 0,
        count: int = 10,
        queue: Optional[int] = 420,
    ) -> list[str]:

        base = self._regional_host(regional_routing)
        url = f"{base}/riot/account/v1/accounts/by-riot-id/{puuid}"
        params: dict[str, Any] = {"start": start, "count": count}

        if queue is not None:
            params["queue"] = queue

        return self._get(url, params=params)

    def get_match(self, match_id: str, regional_routing: str) -> dict[str, Any]:
        base = self._regional_host(regional_routing)
        url = f"{base}/lol/match/v5/matches/{match_id}"
        return self._get(url)

    def get_timeline(self, match_id: str, regional_routing: str) -> dict[str, Any]:
        base = self._regional_host(regional_routing)
        url = f"{base}/lol/match/v5/matches/{match_id}/timeline"
        return self._get(url)
