"""Every sizeable call decision in the arena replays, with the seat's real equity: the data behind the price card.

    uv run python price_spots.py --replays <dir of replay .json> --out runs/price_spots.jsonl

`puzzles.py` keeps only stack-sized calls and river bets. This keeps every decision where a seat faced a bet worth at
least `--min-share` of its stack (or any river bet), from every seat's point of view, and records what the seat could
see (cards, price, share of stack, players left, opponents still in, whether a bettor is all-in) next to its equity
against the hands that stayed in. `opsd/price_card.py` turns these into win-rate tables.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from clone_from_replays import replay_hand
from puzzles import equity

SELECT_ONE_IN = 6  # the same held-out matches as rl/make_puzzle_dataset.py


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
                stack_before = example["stack_after"] + (to_call if took == "call" else 0)
                if took in ("bet", "raise"):
                    stack_before = max(stack_before, to_call)
                share = to_call / max(1, stack_before)
                if share < min_share and example["street"] != "river":
                    continue
                folded_before = {a["seat"] for a in hand["actions"][:index] if a["action"] == "fold"}
                folded_after = {a["seat"] for a in hand["actions"][index + 1 :] if a["action"] == "fold"}
                villains = [hole[s] for s in hole if s != seat and s not in folded_before and s not in folded_after]
                if not villains:
                    continue
                seed = int(hashlib.sha1(f"{match['id']}:{hand['hand_number']}:{seat}:{index}".encode()).hexdigest(), 16) & 0xFFFF
                out.append({
                    "id": f"{match['id']}:{hand['hand_number']}:{seat}:{index}", "held_out": held_out(match["id"]),
                    "entry": entry["name"], "street": example["street"], "hole": example["hole"], "board": example["board"],
                    "to_call": to_call, "pot": pot, "stack": stack_before, "share": round(share, 3),
                    "players_left": players_left, "in_hand": len(hole) - len(folded_before) - 1, "villains": len(villains),
                    "equity": round(equity(example["hole"], villains, example["board"], seed), 3),
                    "took": "fold" if took == "fold" else "continue", "messages": example["messages"],
                })
    return out


def one_file(job: tuple[str, float]) -> list[dict]:
    return asyncio.run(spots(Path(job[0]), job[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replays", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("runs/price_spots.jsonl"))
    parser.add_argument("--min-share", type=float, default=0.12, help="smallest call, as a share of the seat's stack")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    jobs = [(str(p), args.min_share) for p in sorted(args.replays.glob("*.json"))]
    rows: list[dict] = []
    with ProcessPoolExecutor(args.workers) as pool:
        for found in pool.map(one_file, jobs):
            rows.extend(found)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} call decisions from {len(jobs)} replays written to {args.out} ({sum(r['held_out'] for r in rows)} held out)")


if __name__ == "__main__":
    main()
