"""Derive training and evaluation opponents from public, completed arena replays."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

FIELD = "field"
HOUSE_ENTRIES = {
    "house-anxious-alice": "house_alice",
    "house-bob-the-buster": "house_bob",
    "house-stubborn-stan": "house_stan",
    "house-chatty-carlos": "house_carlos",
    "house-professor-pete": "house_pete",
    "house-tilted-tim": "house_tim",
    "house:nit": "nit",
    "house:maniac": "maniac",
    "house:calling-station": "calling_station",
    "house:random": "random",
}


@dataclass(frozen=True)
class ArenaRoster:
    match_id: str
    hero: int
    seats: tuple[str, ...]

    def opponent_specs(self, stacks: Sequence[int]) -> dict[str, str | dict]:
        """Whole-hand rollout specs for seats still in the game at the snapshot."""
        if len(stacks) != len(self.seats):
            raise ValueError("snapshot and replay use different table sizes")
        return {
            str(seat): {"thinking": False} if style == FIELD else f"scripted:{style}"
            for seat, (style, stack) in enumerate(zip(self.seats, stacks, strict=True))
            if seat != self.hero and stack > 0
        }


def replay_roster(match: dict, entry: str) -> ArenaRoster | None:
    if not match.get("finished_at"):
        return None
    seats = {row["seat"]: row["entry"] for row in match["seats"]}
    hero = [
        seat
        for seat, info in seats.items()
        if info.get("kind") == "team" and entry in (info.get("id"), info.get("org_id"), info.get("name"))
    ]
    if not hero:
        return None
    if len(hero) != 1 or set(seats) != set(range(6)):
        raise ValueError(f"{match['id']}: expected one policy seat at a six-seat table")
    styles = []
    for seat in range(6):
        info = seats[seat]
        if seat == hero[0]:
            styles.append("model")
        elif info.get("kind") == "house":
            if info["id"] not in HOUSE_ENTRIES:
                raise ValueError(f"{match['id']}: unknown house entry {info['id']}")
            styles.append(HOUSE_ENTRIES[info["id"]])
        else:
            styles.append(FIELD)
    return ArenaRoster(match["id"], hero[0], tuple(styles))


def recent_rosters(directory: Path, entry: str, limit: int = 50) -> tuple[ArenaRoster, ...]:
    matches = {}
    for path in directory.glob("*.json"):
        match = json.loads(path.read_text())
        if match.get("kind") == "ranked" and (roster := replay_roster(match, entry)):
            matches[roster.match_id] = (match["finished_at"], roster)
    if not matches:
        raise ValueError(f"no finished ranked tables for {entry} in {directory}")
    latest = sorted(matches.values(), key=lambda row: (row[0], row[1].match_id), reverse=True)
    return tuple(roster for _, roster in latest[:limit])
