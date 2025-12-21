from __future__ import annotations
import time

from dotenv import load_dotenv

from cache import JsonCache
from riot_client import RiotClient

DEFAULT_REGIONAL_ROUTING = "americas"


def main():
    load_dotenv()

    riot_id = "Ziad"
    tag_line = "OG09"
    regional = DEFAULT_REGIONAL_ROUTING

    match_count = 10
    queue = 420  # Ranked Solo/Duo

    client = RiotClient()
    cache = JsonCache()

    puuid = client.get_puuid_by_riot_id(riot_id, tag_line, regional_routing=regional)
    print(f"PUUID: {puuid}")

    match_ids = client.get_match_ids_by_puuid(
        puuid, regional_routing=regional, start=0, count=match_count, queue=queue
    )
    print(f"Found {len(match_ids)} match ids")

    for i, match_id in enumerate(match_ids, start=1):
        # Match details cache
        if cache.get("match", match_id) is None:
            match = client.get_match(match_id, regional_routing=regional)
            cache.set("match", match_id, match)
            print(f"[{i}/{len(match_ids)}] saved match {match_id}")
        else:
            print(f"[{i}/{len(match_ids)}] match cached {match_id}")

        # Timeline cache
        if cache.get("timeline", match_id) is None:
            timeline = client.get_timeline(match_id, regional_routing=regional)
            cache.set("timeline", match_id, timeline)
            print(f"[{i}/{len(match_ids)}] saved timeline {match_id}")
        else:
            print(f"[{i}/{len(match_ids)}] timeline cached {match_id}")

        time.sleep(0.25)

    print("Done. Cached files are in data/cache/matches and data/cache/timelines.")


if __name__ == "__main__":
    main()
