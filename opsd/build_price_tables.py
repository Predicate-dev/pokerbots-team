"""Measure the price card's tables from arena replays, and test the card on held-out matches.

    uv run python opsd/build_price_tables.py --replays <dir of replay .json> --spots harness/runs/price_spots.jsonl

Writes `opsd/price_tables.json`:
- `value["<players left>|<stage>"]`: [[stack, rating points], ...], the average rating change of seats that held about
  that stack with that many players left (stage 0/1/2 = hands 1-10, 11-20, 21-30), made non-decreasing in the stack.
- `win["<spot key>"]`: [average real equity, count] for each kind of hand in each kind of spot (`price_card.spot_keys`).

Matches whose id hashes to the held-out sixth (the same split as the puzzle datasets) are left out of both tables.
The report then scores, on those held-out matches, three ways of answering each call decision: what the seat did,
the chip price alone, and the card. The score is in rating points per decision, using each spot's real equity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import price_card  # noqa: E402
from price_card import PLACE_POINTS, card, parse, spot_keys, stack_value  # noqa: E402

BINS = [1, 60, 120, 180, 230, 320, 450, 700, 100_000]
MIN_CELL = 8


def held_out(match_id: str) -> bool:
    return int(hashlib.sha1(match_id.encode()).hexdigest(), 16) % 6 == 0


def value_table(replays: Path) -> dict[str, list[list[float]]]:
    cells: dict[tuple[int, int, int], list[tuple[int, float]]] = defaultdict(list)
    for path in sorted(replays.glob("*.json")):
        match = json.loads(path.read_text())
        if match["kind"] != "ranked" or held_out(match["id"]):
            continue
        place = {r["seat"]: r["place"] for r in match["results"]}
        for hand in match["hands"]:
            stacks = {int(seat): chips for seat, chips in hand["stacks_after"].items() if chips > 0}
            stage = 0 if hand["hand_number"] <= 10 else 1 if hand["hand_number"] <= 20 else 2
            for seat, chips in stacks.items():
                b = max(i for i, low in enumerate(BINS[:-1]) if chips >= low)
                cells[(len(stacks), stage, b)].append((chips, PLACE_POINTS[place[seat]]))
    table = {}
    for live in range(2, 7):
        for stage in range(3):
            points = []
            for b in range(len(BINS) - 1):
                rows = cells.get((live, stage, b), [])
                if len(rows) < MIN_CELL:  # too few at this stage: pool the stages
                    rows = [r for s in range(3) for r in cells.get((live, s, b), [])]
                if len(rows) >= MIN_CELL:
                    points.append([statistics.mean(r[0] for r in rows), statistics.mean(r[1] for r in rows), len(rows)])
            floor = PLACE_POINTS[live]
            ys = [max(floor, min(PLACE_POINTS[1], p[1])) for p in points]
            # pool adjacent violators: a bigger stack is never worth less
            blocks = [[y, p[2]] for y, p in zip(ys, points, strict=True)]
            sizes = [1] * len(blocks)
            i = 0
            while i < len(blocks) - 1:
                if blocks[i][0] > blocks[i + 1][0]:
                    total = blocks[i][1] + blocks[i + 1][1]
                    blocks[i] = [(blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]) / total, total]
                    sizes[i] += sizes[i + 1]
                    del blocks[i + 1], sizes[i + 1]
                    i = max(0, i - 1)
                else:
                    i += 1
            smooth = [block[0] for block, size in zip(blocks, sizes, strict=True) for _ in range(size)]
            table[f"{live}|{stage}"] = [[round(p[0], 1), round(y, 3)] for p, y in zip(points, smooth, strict=True)]
    return table


def win_table(spots: list[dict]) -> dict[str, list[float]]:
    cells: dict[str, list[float]] = defaultdict(list)
    for row in spots:
        spot = parse(row["messages"])
        if spot is None:
            continue
        for key in spot_keys(spot):
            cells[key].append(row["equity"])
    return {key: [round(statistics.mean(v), 4), len(v)] for key, v in sorted(cells.items())}


def gain(row: dict, c: dict, answer: str) -> float:
    """Rating points the answer is worth over folding, from the spot's real equity."""
    if answer != "continue":
        return 0.0
    return row["equity"] * c["win_value"] + (1 - row["equity"]) * c["lose_value"] - c["fold_value"]


def report(name: str, spots: list[dict], tables: dict) -> None:
    rows = [(row, c) for row in spots if (c := card(row["messages"], tables)) is not None]
    if not rows:
        return
    print(f"\n{name}: {len(rows)} call decisions")
    answers = {
        "what the seat did": lambda row, c: row["took"],
        "always fold": lambda row, c: "fold",
        "chip price alone": lambda row, c: "continue" if c["win"] > c["price"] else "fold",
        "the price card": lambda row, c: "continue" if c["win"] > c["need"] else "fold",
        "perfect hindsight": lambda row, c: "continue" if gain(row, c, "continue") > 0 else "fold",
    }
    for group, keep in (
        ("all", lambda row, c: True), ("preflop", lambda row, c: c["street"] == "preflop"),
        ("postflop", lambda row, c: c["street"] != "preflop"),
        ("calls for 40%+ of the stack", lambda row, c: c["share"] >= 0.4),
        ("5-6 players left", lambda row, c: c["live"] >= 5), ("4 or fewer left", lambda row, c: c["live"] <= 4),
        ("our own decisions", lambda row, c: row["entry"] == "Friends"),
    ):
        part = [(row, c) for row, c in rows if keep(row, c)]
        if not part:
            continue
        print(f"  {group} ({len(part)})")
        for label, answer in answers.items():
            total = sum(gain(row, c, answer(row, c)) for row, c in part)
            right = sum((answer(row, c) == "continue") == (gain(row, c, "continue") > 0) for row, c in part)
            share = sum(answer(row, c) == "continue" for row, c in part) / len(part)
            print(f"    {label:18} {1000 * total / len(part):+7.1f} rating points per 1000 decisions | best answer "
                  f"{100 * right / len(part):3.0f}% | continues {100 * share:3.0f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replays", type=Path, required=True)
    parser.add_argument("--spots", type=Path, required=True, help="from harness/price_spots.py")
    parser.add_argument("--out", type=Path, default=price_card.TABLES)
    args = parser.parse_args()
    spots = [json.loads(line) for line in args.spots.read_text().splitlines() if line.strip()]
    train = [s for s in spots if not s["held_out"]]
    test = [s for s in spots if s["held_out"]]
    tables = {"value": value_table(args.replays), "win": win_table(train)}
    args.out.write_text(json.dumps(tables, indent=1))
    print(f"wrote {args.out}: {len(tables['value'])} value curves, {len(tables['win'])} win-rate cells from {len(train)} decisions")
    for live in (6, 5, 4, 3):
        print(f"  {live} left, hands 1-10: " + ", ".join(f"{s} chips {stack_value(live, 5, s, tables['value']):+.2f}" for s in (0, 50, 100, 200, 300, 400, 600)))
    report("matches the tables were measured on", train, tables)
    report("held-out matches", test, tables)


if __name__ == "__main__":
    main()
