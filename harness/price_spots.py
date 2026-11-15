"""Extract call and first-in opening decisions from the arena's public replays.

    uv run python price_spots.py --replays <dir of replay .json> --out runs/price_spots.jsonl
    uv run python price_spots.py --replays <dir of replay .json> --kind open --out runs/open_spots.jsonl

`puzzles.py` keeps only stack-sized calls and river bets. This keeps every decision where a seat faced a bet worth at
least `--min-share` of its stack (or any river bet), from every seat's point of view, and records what the seat could
see (cards, price, share of stack, players left, opponents still in, whether a bettor is all-in) next to its equity
against the hands that stayed in. `opsd/price_card.py` turns these into win-rate tables.

The open mode keeps the first preflop decisions up to and including the first voluntary entry. It records the
actual action and the before/after stacks and live player counts for measuring rating value.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from clone_from_replays import replay_hand
from puzzles import equity
from replay_chips import stack_before

SELECT_ONE_IN = 6  # the same held-out matches as rl/make_puzzle_dataset.py
FORCED = ("fallback", "coerced", "format_error", "afk", "timed_out", "token_cap", "sat_out", "error", "scripted")


def held_out(match_id: str) -> bool:
    return int(hashlib.sha1(match_id.encode()).hexdigest(), 16) % SELECT_ONE_IN == 0


async def spots(path: Path, min_share: float) -> list[dict]:
    match = json.loads(path.read_text())
    if match["kind"] != "ranked":
        return []
    out = []
    entries = {s["seat"]: s["entry"] for s in match["seats"]}
    for seat, entry in entries.items():
        for hand in match["hands"]:
            if seat not in {p["seat"] for p in hand["players"]}:
                continue
            if not any(a["seat"] == seat and a["to_call"] > 0 for a in hand["actions"]):
                continue
            try:
                examples, problem, _, _ = await replay_hand(match, hand, seat)
            except Exception:  # noqa: BLE001 - a hand that can't be replayed is skipped
                problem = "error"
            if problem:
                continue
            hole = {p["seat"]: p["hole_cards"] for p in hand["players"]}
            players_left = sum(p["starting_stack"] > 0 for p in hand["players"])
            for example in examples:
                to_call, pot, index = example["to_call"], example["pot_before"], example["action_index"]
                if to_call <= 0:
                    continue
                took = example["action"]["action"]
                stack = stack_before(hand["actions"][index])
                share = to_call / max(1, stack)
                if share < min_share and example["street"] != "river":
                    continue
                folded_before = {a["seat"] for a in hand["actions"][:index] if a["action"] == "fold"}
                folded_after = {a["seat"] for a in hand["actions"][index + 1 :] if a["action"] == "fold"}
                villains = [hole[s] for s in hole if s != seat and s not in folded_before and s not in folded_after]
                if not villains:
                    continue
                seed = int(hashlib.sha1(f"{match['id']}:{hand['hand_number']}:{seat}:{index}".encode()).hexdigest(), 16) & 0xFFFF
                raises = [a for a in hand["actions"][:index] if a["action"] in ("bet", "raise") and a["seat"] != seat]
                bettor = entries[raises[-1]["seat"]] if raises else None  # who made the bet the seat is facing
                out.append({
                    "bettor": bettor["name"] if bettor else None, "bettor_kind": bettor["kind"] if bettor else None,
                    "bettor_said": [a["say"] for a in hand["actions"][:index] if raises and a["seat"] == raises[-1]["seat"] and a.get("say")],
                    "id": f"{match['id']}:{hand['hand_number']}:{seat}:{index}", "held_out": held_out(match["id"]),
                    "entry": entry["name"], "street": example["street"], "hole": example["hole"], "board": example["board"],
                    "to_call": to_call, "pot": pot, "stack": stack, "share": round(share, 3),
                    "players_left": players_left, "in_hand": len(hole) - len(folded_before) - 1, "villains": len(villains),
                    "equity": round(equity(example["hole"], villains, example["board"], seed), 3),
                    "took": "fold" if took == "fold" else "continue", "messages": example["messages"],
                })
    return out


def first_in_actions(actions: list[dict]) -> Iterator[tuple[int, dict]]:
    """Yield decisions facing the blinds until a player voluntarily enters the pot."""
    for index, action in enumerate(actions):
        if action["street"] != "preflop":
            break
        if action["to_call"] > 0:
            yield index, action
        if action["action"] != "fold":
            break


async def opening_spots(path: Path) -> list[dict]:
    match = json.loads(path.read_text())
    if match["kind"] != "ranked":
        return []
    match_id = match["id"]
    entries = {s["seat"]: s["entry"]["name"] for s in match["seats"]}
    places = {r["seat"]: r["place"] for r in match["results"]}
    out = []
    for hand in match["hands"]:
        starting = {p["seat"]: p["starting_stack"] for p in hand["players"] if p["starting_stack"] > 0}
        ending = {int(seat): chips for seat, chips in hand["stacks_after"].items() if chips > 0}
        for index, action in first_in_actions(hand["actions"]):
            seat = action["seat"]
            examples, problem, _, _ = await replay_hand(match, hand, seat, include_forced=True)
            if problem:
                continue
            example = next((e for e in examples if e["action_index"] == index), None)
            if example is None:
                continue
            out.append({
                "id": f"{match_id}:{hand['hand_number']}:{seat}:{index}",
                "held_out": held_out(match_id),
                "entry": entries[seat],
                "took": action["action"],
                "forced": any(action.get(flag) for flag in FORCED),
                "raise_to": action["amount"] if action["action"] == "raise" else None,
                "stack_before": starting[seat],
                "stack_after": ending.get(seat, 0),
                "live_after": len(ending),
                "bust_place": None if seat in ending else places[seat],
                "messages": example["messages"],
            })
    return out


def one_file(job: tuple[str, str, float]) -> list[dict]:
    path, kind, min_share = job
    return asyncio.run(opening_spots(Path(path)) if kind == "open" else spots(Path(path), min_share))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replays", type=Path, required=True)
    parser.add_argument("--kind", choices=("call", "open"), default="call")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--min-share", type=float, default=0.12, help="smallest call, as a share of the seat's stack")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    out = args.out or Path("runs/open_spots.jsonl" if args.kind == "open" else "runs/price_spots.jsonl")
    jobs = [(str(p), args.kind, args.min_share) for p in sorted(args.replays.glob("*.json"))]
    if not jobs:
        parser.error(f"no replay files found under {args.replays}")
    rows: list[dict] = []
    with ProcessPoolExecutor(args.workers) as pool:
        for found in pool.map(one_file, jobs):
            rows.extend(found)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} {args.kind} decisions from {len(jobs)} replays written to {out} "
          f"({sum(r['held_out'] for r in rows)} held out)")


if __name__ == "__main__":
    main()
