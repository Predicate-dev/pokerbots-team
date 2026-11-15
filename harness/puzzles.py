"""A puzzle bank: real decision spots from arena replays, each with the answer the cards say was best.

    uv run python puzzles.py build --replays <dir of replay .json> --out runs/puzzles.jsonl [--team "Friends"]
    AC2_API_KEY=<key> uv run python puzzles.py run --puzzles runs/puzzles.jsonl --model vals-hackathon/<deployment>

A full game takes about 20 minutes and gives one noisy result. A puzzle is one decision: the exact official prompt a
seat saw (rebuilt by `clone_from_replays.py`), sent to a deployment once. A few hundred of them run in minutes, so a
new checkpoint can be checked before it is deployed to the arena.

Only spots whose answer the replay settles are used, because replays show every seat's cards:

    stack_call   facing a bet that costs at least 40% of the seat's stack (or all of it). The seat's equity against
                 the hands that actually stayed in is computed by dealing out the rest of the board. The best answer
                 is to continue when that equity beats the price by a safety margin, else fold. The margin is larger
                 while five or six players remain, because busting then means finishing last or next to last.
    river_bet    facing any other bet on the river. The best answer is to call when the seat's hand wins the
                 showdown against the hands that stayed in, else fold.

These are hindsight answers against the real cards, not a strategy against a range, so a single puzzle is not proof
of a mistake. Across a few hundred they measure what matters here: does the model put its stack in ahead or behind,
and does it pay off or fold winners on the river. `regret` is the chip EV the wrong answer gives up.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
from collections import Counter
from itertools import combinations
from pathlib import Path

import httpx
from pokerkit import Card, Deck, StandardHighHand

from clone_from_replays import replay_hand
from poker_harness.completions import GATEWAY_URL, relay_completion, with_preset
from poker_harness.prompts import talk_fields

STACK_SHARE = 0.4  # a call for at least this share of the seat's stack is a stack decision
MARGIN_EARLY, MARGIN_LATE = 0.08, 0.0  # extra equity wanted over the price with 5-6 players left, and with fewer
SAMPLES = 400


def _cards(codes: list[str]) -> tuple[Card, ...]:
    return tuple(next(iter(Card.parse(code))) for code in codes)


def equity(hole: list[str], villains: list[list[str]], board: list[str], seed: int) -> float:
    """The seat's share of the pot at showdown against these exact hands, over the rest of the board."""
    used = set(hole) | {c for v in villains for c in v} | set(board)
    deck = [repr(card) for card in Deck.STANDARD if repr(card) not in used]
    need = 5 - len(board)
    if need == 0:
        runouts = [()]
    elif need <= 2:
        runouts = list(combinations(deck, need))
    else:
        rng = random.Random(seed)
        runouts = [tuple(rng.sample(deck, need)) for _ in range(SAMPLES)]
    mine, theirs = _cards(hole), [_cards(v) for v in villains]
    share = 0.0
    for extra in runouts:
        full = _cards([*board, *extra])
        best = StandardHighHand.from_game(mine, full)
        others = [StandardHighHand.from_game(v, full) for v in theirs]
        top = max(others)
        if best > top:
            share += 1.0
        elif best == top:
            share += 1.0 / (1 + sum(o == top for o in others))
    return share / len(runouts)


def label(example: dict, hand: dict, players_left: int, seed: int) -> dict | None:
    """The puzzle for one decision, or None when the replay does not settle its answer."""
    to_call, pot = example["to_call"], example["pot_before"]
    if to_call <= 0:
        return None
    hero, index = example["seat"], example["action_index"]
    hole = {p["seat"]: p["hole_cards"] for p in hand["players"]}
    folded_before = {a["seat"] for a in hand["actions"][:index] if a["action"] == "fold"}
    folded_after = {a["seat"] for a in hand["actions"][index + 1 :] if a["action"] == "fold"}
    villains = [hole[s] for s in hole if s != hero and s not in folded_before and s not in folded_after]
    if not villains:
        return None
    took = example["action"]["action"]
    stack_before = example["stack_after"] + (to_call if took == "call" else 0) if took != "fold" else None
    if stack_before is None:  # a fold leaves the stack as it was
        stack_before = example["stack_after"]
    share = to_call / max(1, stack_before)
    price = to_call / (pot + to_call)
    eq = equity(example["hole"], villains, example["board"], seed)
    if share >= STACK_SHARE:
        kind = "stack_call"
        margin = MARGIN_EARLY if players_left >= 5 else MARGIN_LATE
        best = "continue" if eq >= price + margin else "fold"
    elif example["street"] == "river":
        kind, margin = "river_bet", 0.0
        best = "continue" if eq >= 0.5 else "fold"
    else:
        return None
    ev_continue = eq * (pot + to_call) - to_call
    return {
        "kind": kind, "best": best, "equity": round(eq, 3), "price": round(price, 3), "margin": margin,
        "regret_if_wrong": round(abs(ev_continue), 1), "players_left": players_left, "villains": len(villains),
        "took": "fold" if took == "fold" else "continue",
    }


async def build(args: argparse.Namespace) -> None:
    out, stats = [], Counter()
    for path in sorted(args.replays.glob("*.json")):
        match = json.loads(path.read_text())
        if match["kind"] != "ranked":
            continue
        entries = {s["seat"]: s["entry"] for s in match["seats"]}
        for seat, entry in entries.items():
            ours = entry["name"] == args.team
            if args.only_team and not ours:
                continue
            for hand in match["hands"]:
                dealt = {p["seat"] for p in hand["players"]}
                if seat not in dealt or not any(a["seat"] == seat and a["to_call"] > 0 for a in hand["actions"]):
                    continue
                try:
                    examples, problem, _, _ = await replay_hand(match, hand, seat)
                except Exception:  # noqa: BLE001 - a hand that can't be replayed is skipped and counted
                    examples, problem = [], "error"
                if problem:
                    stats[f"skipped: {problem}"] += 1
                    continue
                players_left = sum(p["starting_stack"] > 0 for p in hand["players"])
                for example in examples:
                    puzzle = label(example, hand, players_left, hash((match["id"], hand["hand_number"], seat)) & 0xFFFF)
                    if puzzle is None:
                        continue
                    stats[puzzle["kind"]] += 1
                    out.append(puzzle | {
                        "id": f"{match['id']}:{hand['hand_number']}:{seat}:{example['action_index']}",
                        "ours": ours, "entry": entry["name"], "deployment": entry.get("deployment_id"),
                        "street": example["street"], "hole": example["hole"], "board": example["board"],
                        "to_call": example["to_call"], "pot": example["pot_before"], "messages": example["messages"],
                    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(p) + "\n" for p in out))
    took_right = sum(p["took"] == p["best"] for p in out)
    print(json.dumps(dict(stats)), f"\n{len(out)} puzzles written to {args.out}; the seats that played them chose the best answer {took_right}/{len(out)}")


async def run(args: argparse.Namespace) -> None:
    key = os.environ.get("AC2_API_KEY")
    if not key:
        raise SystemExit("export AC2_API_KEY first")
    puzzles = [json.loads(line) for line in args.puzzles.read_text().splitlines() if line.strip()]
    if args.only_ours:
        puzzles = [p for p in puzzles if p["ours"]]
    if args.limit:
        puzzles = random.Random(0).sample(puzzles, min(args.limit, len(puzzles)))
    params = with_preset({"preset": "event", "seed": 0, "submit_fields": ["action", "amount"]})
    params["submit_fields"] = talk_fields(params["submit_fields"])
    gate = asyncio.Semaphore(args.parallel)
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=30.0)) as client:
        complete = relay_completion(client, GATEWAY_URL, key, params | {"model_label": args.model})

        async def one(puzzle: dict) -> dict:
            async with gate:
                try:
                    reply = await complete(puzzle["messages"])
                    arguments = reply.tool_arguments
                    parsed = json.loads(arguments) if isinstance(arguments, str) else (arguments or {})
                    action = str(parsed.get("action", "")) if not reply.truncated else ""
                except Exception as exc:  # noqa: BLE001
                    action, parsed = "", {"error": type(exc).__name__}
            answer = "fold" if action == "fold" else "continue" if action in ("call", "raise", "bet") else "invalid"
            return puzzle | {"answer": answer, "say": parsed.get("say", "") if isinstance(parsed, dict) else ""}

        rows = list(await asyncio.gather(*(one(p) for p in puzzles)))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("".join(json.dumps({k: v for k, v in r.items() if k != "messages"}) + "\n" for r in rows))

    def line(name: str, group: list[dict]) -> str:
        if not group:
            return f"  {name:22} none"
        right = sum(r["answer"] == r["best"] for r in group)
        regret = sum(r["regret_if_wrong"] for r in group if r["answer"] != r["best"]) / len(group)
        bad_calls = sum(r["answer"] == "continue" and r["best"] == "fold" for r in group)
        bad_folds = sum(r["answer"] == "fold" and r["best"] == "continue" for r in group)
        return (f"  {name:22} {right}/{len(group)} best ({100 * right / len(group):.0f}%) | chips given up per puzzle "
                f"{regret:.1f} | called when behind {bad_calls} | folded when ahead {bad_folds} | "
                f"invalid {sum(r['answer'] == 'invalid' for r in group)}")

    print(f"{args.model}: {len(rows)} puzzles")
    print(line("all", rows))
    for kind in ("stack_call", "river_bet"):
        print(line(kind, [r for r in rows if r["kind"] == kind]))
    print(line("stack_call, 5-6 left", [r for r in rows if r["kind"] == "stack_call" and r["players_left"] >= 5]))
    print(line("from our own games", [r for r in rows if r["ours"]]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--replays", type=Path, required=True)
    b.add_argument("--out", type=Path, default=Path("runs/puzzles.jsonl"))
    b.add_argument("--team", default="Friends", help="entry name whose spots are marked as ours")
    b.add_argument("--only-team", action="store_true", help="only this team's decisions (default: every seat's)")
    r = sub.add_parser("run")
    r.add_argument("--puzzles", type=Path, default=Path("runs/puzzles.jsonl"))
    r.add_argument("--model", required=True)
    r.add_argument("--parallel", type=int, default=24)
    r.add_argument("--limit", type=int, help="a fixed random subset of this many puzzles")
    r.add_argument("--only-ours", action="store_true")
    r.add_argument("--out", type=Path)
    args = parser.parse_args()
    asyncio.run(build(args) if args.command == "build" else run(args))


if __name__ == "__main__":
    main()
