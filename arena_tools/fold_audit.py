"""Report intentional folds by checkpoint, street and preflop situation in public ranked replays."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import fmean

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness" / "src"))
from poker_harness.placement import rating_points  # noqa: E402
from poker_harness.preflop import PreflopAction, opportunities, record_folds  # noqa: E402

_NON_POLICY = ("fallback", "coerced", "format_error", "afk", "timed_out", "token_cap", "sat_out", "error", "scripted")


def audit(directory: Path, org: str) -> list[dict]:
    matches = {}
    for path in directory.glob("*.json"):
        match = json.loads(path.read_text())
        if match.get("schema") == "vals-arena.match/1" and match["kind"] == "ranked" and match.get("finished_at"):
            matches[match["id"]] = match
    versions = {}
    for match in sorted(matches.values(), key=lambda match: match["finished_at"]):
        player = next((row for row in match["seats"] if row["entry"].get("org_id") == org), None)
        if player is None:
            continue
        hero = player["seat"]
        entry = player["entry"]
        deployment = entry.get("deployment") or {}
        identity = (
            entry["submission_name"],
            entry["deployment_id"],
            deployment.get("version"),
            deployment.get("checkpoint_id"),
        )
        version = versions.setdefault(
            identity,
            {
                "submission": identity[0],
                "deployment": identity[1],
                "deployment_version": identity[2],
                "checkpoint": identity[3],
                "first_finished_at": match["finished_at"],
                "places": Counter(),
                "rating_points_total": 0.0,
                "busts": 0,
                "decisions": 0,
                "forced_or_coerced": 0,
                "hands_dealt": 0,
                "vpip_hands": 0,
                "priced_by_street": {},
                "preflop_spots": {},
            },
        )
        version["last_finished_at"] = match["finished_at"]
        result = next(result for result in match["results"] if result["seat"] == hero)
        version["places"][result["place"]] += 1
        version["rating_points_total"] += rating_points(result["place"], len(match["seats"]))
        version["busts"] += result.get("eliminated_on_hand") is not None
        for hand in match["hands"]:
            seats = {row["seat"] for row in hand["players"]}
            if hero not in seats:
                continue
            version["hands_dealt"] += 1
            preflop = [action for action in hand["actions"] if action["street"] == "preflop"]
            moves = (PreflopAction(a["seat"], a["action"], a["to_call"], a["amount"]) for a in preflop)
            spots = opportunities(moves, seats, hand["button_seat"], hand["big_blind"])
            vpip = False
            for action, tags in zip(preflop, spots, strict=True):
                if action["seat"] == hero and not any(action.get(flag) for flag in _NON_POLICY):
                    kind = action["action"]
                    vpip |= kind in ("bet", "raise") or (kind == "call" and action["to_call"] > 0)
                    record_folds(version["preflop_spots"], tags, kind)
            version["vpip_hands"] += vpip
            for action in hand["actions"]:
                if action["seat"] != hero:
                    continue
                version["decisions"] += 1
                if any(action.get(flag) for flag in _NON_POLICY):
                    version["forced_or_coerced"] += 1
                elif action["to_call"] > 0:
                    record_folds(version["priced_by_street"], (action["street"],), action["action"])
    report = []
    for version in versions.values():
        games = sum(version["places"].values())
        points = version.pop("rating_points_total")
        report.append(version | {
            "games": games,
            "mean_place": round(fmean(version["places"].elements()), 3),
            "rating_points_per_game": round(points / games, 3),
        })
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replays", type=Path, help="directory of public replay JSON")
    parser.add_argument("--org", required=True, help="team org ID from the replay")
    args = parser.parse_args()
    report = audit(args.replays, args.org)
    if not report:
        parser.error(f"no completed ranked replays for {args.org} in {args.replays}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
