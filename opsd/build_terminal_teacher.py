"""Build legal call/fold SFT examples and compare the teacher on held-out arena replays."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path
from statistics import fmean

import price_card
import terminal_calls as terminal
from poker_harness.completions import EVENT_PARAMS, table_rules
from poker_harness.prompts import system_prompt
from pokerkit import HandHistory

FORCED = (
    "fallback",
    "coerced",
    "format_error",
    "afk",
    "timed_out",
    "token_cap",
    "sat_out",
    "error",
    "scripted",
)
OFFICIAL_SYSTEM = {
    "role": "system",
    "content": system_prompt(table_rules(EVENT_PARAMS)),
}


def held_out(identity: str) -> bool:
    return int(hashlib.sha1(identity.encode()).hexdigest(), 16) % 6 == 0


def pluribus_counts(directory: Path) -> dict:
    counts = defaultdict(lambda: [0, 0])
    for path in sorted(directory.rglob("*.phh")):
        if held_out(str(path.parent.relative_to(directory))):
            continue
        with path.open("rb") as src:
            hand = HandHistory.load(src)
        hero = hand.players.index("Pluribus")
        pending = None
        for state, action in hand.state_actions:
            if pending and action:
                parts = action.split()
                if parts[0] == f"p{hero + 1}" and parts[1] in ("cbr", "cc", "f"):
                    counts[pending][0] += 1
                    counts[pending][1] += parts[1] == "cbr"
            pending = None
            if state.actor_index != hero or not state.can_complete_bet_or_raise_to():
                continue
            street = ("preflop", "flop", "turn", "river")[state.street_index]
            facing = max(state.bets) > (100 if street == "preflop" else 0)
            hole = tuple(map(repr, state.hole_cards[hero]))
            board = tuple(map(repr, state.get_board_cards(0)))
            pending, _ = terminal.context(
                street, facing, False, terminal.hand_bucket(hole, board)
            )
    return dict(counts)


def notes(hand: dict, counts: dict | None = None) -> dict[int, tuple[int, bool]]:
    """Record the last bettor and whether they were facing a bet at every arena decision."""
    starting = {p["seat"]: p["starting_stack"] for p in hand["players"]}
    holes = {p["seat"]: tuple(p["hole_cards"]) for p in hand["players"]}
    bets = {seat: hand["blinds"].get(str(seat), 0) for seat in starting}
    remaining = {seat: chips - bets[seat] for seat, chips in starting.items()}
    active = set(starting)
    street, aggressor = "preflop", None
    before = {}
    for i, a in enumerate(hand["actions"]):
        if a["street"] != street:
            street, aggressor = a["street"], None
            bets = dict.fromkeys(starting, 0)
        if aggressor is not None:
            before[i] = aggressor
        facing = max(bets.values()) > (hand["big_blind"] if street == "preflop" else 0)
        seat, action = a["seat"], a["action"]
        aggressive = action in ("bet", "raise")
        put_in = (
            (a["amount"] - bets[seat])
            if aggressive
            else a["to_call"]
            if action == "call"
            else 0
        )
        can_raise = remaining[seat] > a["to_call"] and any(
            remaining[s] > 0 for s in active - {seat}
        )
        if counts is not None and can_raise and not any(a.get(flag) for flag in FORCED):
            bucket = terminal.hand_bucket(holes[seat], tuple(a["board"]))
            late = terminal.late_position(seat, hand["button_seat"], starting)
            for key in terminal.context(street, facing, late, bucket):
                cell = counts[key]
                cell[0] += 1
                cell[1] += aggressive
                cell[2] += aggressive and put_in >= remaining[seat]
        remaining[seat] -= put_in
        bets[seat] += put_in
        if aggressive:
            aggressor = (seat, facing)
        if action == "fold":
            active.remove(seat)
    return before


def load(path: Path) -> list[dict]:
    with path.open() as file:
        return [json.loads(line) for line in file if line.strip()]


def archive_spots(args, rows: list[dict]) -> tuple[dict, list, Counter]:
    by_match = defaultdict(list)
    for row in rows:
        by_match[row["id"].split(":", 1)[0]].append(row)
    counts = defaultdict(lambda: [0, 0, 0])
    examples, stats = [], Counter()
    for match_id, items in by_match.items():
        match = json.loads((args.replays / f"{match_id}.json").read_text())
        hands = {h["hand_number"]: h for h in match["hands"]}
        train = items[0]["split"] == "train"
        betting = {n: notes(h, counts if train else None) for n, h in hands.items()}
        for row in items:
            if not row["messages"] or row["messages"][0] != OFFICIAL_SYSTEM:
                stats["nonofficial_prompt"] += 1
                continue
            _, number, _, index = row["id"].split(":")
            number, index = int(number), int(index)
            hand, line = hands[number], betting[number].get(index)
            a = hand["actions"][index]
            if (
                line is None
                or a["action"] not in ("fold", "call")
                or any(a.get(flag) for flag in FORCED)
            ):
                continue
            starting = {p["seat"]: p["starting_stack"] for p in hand["players"]}
            spot = terminal.public_call(row["messages"], starting, *line)
            if spot is None:
                continue
            villain = next(
                tuple(p["hole_cards"])
                for p in hand["players"]
                if p["seat"] == spot.villain
            )
            final = tuple(hand["stacks_after"].get(str(s), 0) for s in range(6))
            board = tuple(hand["board"])
            result = (
                None
                if a["action"] == "fold"
                else terminal.outcome(spot, villain, board)
            )
            if terminal.finish_stacks(spot, result) != final:
                stats["payout_mismatch"] += 1
                continue
            stats["payout_verified"] += 1
            if (
                not spot.shove
                and spot.stack[spot.hero] > spot.price
                and terminal.hand_bucket(spot.hole, spot.board)
                in {
                    "monster",
                    "trips",
                    "two_pair",
                }
            ):
                continue  # on a non-all-in river, consider bluff catches and preserve strong value raises
            examples.append((row, spot, villain, a["action"]))
    return dict(counts), examples, stats


def evaluate(job: tuple) -> dict:
    row, spot, villain, took, book, samples = job
    _, gains = terminal.values(spot, price_card.tables()["value"])
    draws = terminal.worlds(spot, samples, row["id"])
    prior = terminal.estimate(spot, draws, book, gains)
    field = terminal.estimate(spot, draws, book, gains, use_pluribus=False)
    old = price_card.card(row["messages"])["plan"]
    old = "call" if old == "continue" else took if old == "close" else "fold"
    action = terminal.robust_action(prior)
    rng = random.Random(row["id"] + ":label")
    deck = [c for c in terminal.DECK if c not in (*spot.hole, *villain, *spot.board)]
    need = 5 - len(spot.board)
    runouts = (
        combinations(deck, need)
        if need <= 1
        else (rng.sample(deck, need) for _ in range(samples))
    )
    realised = fmean(
        gains[terminal.outcome(spot, villain, (*spot.board, *extra))]
        for extra in runouts
    )
    return row | {
        "took": took,
        "street": spot.street,
        "real_gain": realised,
        "robust_action": action,
        "call_cost": spot.price,
        "contestable": terminal.finish_stacks(spot, 1)[spot.hero]
        - spot.stack[spot.hero],
        "range_field_only": terminal.robust_action(field) or old,
        "price_card": old,
        "range_pluribus": action or old,
    }


def report(rows: list[dict]) -> dict:
    if not rows:
        return {}
    policies = ("took", "price_card", "range_field_only", "range_pluribus")
    return {
        "decisions": len(rows),
        "robust_recommendations": sum(row["robust_action"] is not None for row in rows),
        "points_per_1000": {
            policy: round(
                1000
                * fmean(
                    row["real_gain"] if row[policy] == "call" else 0 for row in rows
                ),
                2,
            )
            for policy in policies
        },
        "calls": {p: sum(row[p] == "call" for row in rows) for p in policies},
    }


def sft_example(row: dict) -> dict:
    match, hand, *_ = row["id"].split(":")
    action = row["robust_action"]
    comparison = (
        "the call improves my expected finishing place"
        if action == "call"
        else "folding gives me the better expected finishing place"
    )
    return {
        "messages": row["messages"],
        "action": {"action": action},
        "match": match,
        "hand": int(hand),
        "street": row["street"],
        "reasoning": (
            f"I can call {row['call_cost']} to win {row['contestable']} chips. "
            f"After allowing for refunds, splits and bust risk, {comparison} across plausible betting ranges."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replays", type=Path, required=True)
    parser.add_argument("--pluribus", type=Path, required=True)
    parser.add_argument(
        "--spots",
        type=Path,
        required=True,
        help="Original call bank with hash-based holdout",
    )
    parser.add_argument(
        "--fresh",
        type=Path,
        required=True,
        help="Call spots from disjoint, later matches",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Directory for examples, range book and report",
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--samples", type=int, default=384)
    args = parser.parse_args()
    if args.samples < 1 or args.workers < 1:
        parser.error("--samples and --workers must be positive")
    old, fresh = load(args.spots), load(args.fresh)
    match_ids = lambda rows: {r["id"].split(":", 1)[0] for r in rows}
    if not old or not fresh or match_ids(old) & match_ids(fresh):
        parser.error(
            "the original and fresh files must contain disjoint ranked matches"
        )
    rows = [
        r | {"split": "select" if held_out(r["id"].split(":", 1)[0]) else "train"}
        for r in old
    ]
    rows.extend(r | {"split": "fresh"} for r in fresh)
    arena, spots, stats = archive_spots(args, rows)
    book = {"arena": arena, "pluribus": pluribus_counts(args.pluribus)}
    if not book["pluribus"] or not arena:
        parser.error(
            "both arena training observations and Pluribus actions are required"
        )
    print(
        f"{len(spots)} verified decisions, {sum(n for n, _ in book['pluribus'].values())} Pluribus betting opportunities",
        flush=True,
    )
    with ProcessPoolExecutor(args.workers) as pool:
        scored = list(
            pool.map(
                evaluate, ((*job, book, args.samples) for job in spots), chunksize=8
            )
        )
    train = [r for r in scored if r["split"] == "train" and r["robust_action"]]
    train = [sft_example(row) for row in train]
    result = {
        "payouts": dict(stats),
        "labels": dict(Counter(r["action"]["action"] for r in train)),
        "samples": args.samples,
        **{
            split: report([r for r in scored if r["split"] == split])
            for split in ("train", "select", "fresh")
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "examples.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in train)
    )
    (args.out / "ranges.json").write_text(json.dumps(book, indent=2) + "\n")
    (args.out / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
