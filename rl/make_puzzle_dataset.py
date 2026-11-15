"""Make RL datasets of puzzle tasks: one real decision per rollout, rewarded from the cards (`puzzle_tasks.py`).

    (cd harness && uv run python puzzles.py build --replays ../arena_data --out runs/puzzles.jsonl)
    uv run python rl/make_puzzle_dataset.py --puzzles harness/runs/puzzles.jsonl --replays arena_data            # dry run
    uv run python rl/make_puzzle_dataset.py --puzzles harness/runs/puzzles.jsonl --replays arena_data --upload
    uv run python rl/train.py --prefix pokerbots-puzzle --from-checkpoint <ckpt> --max-response-len 8192 --dry-run

    # rating-scored and balanced: every sizeable call decision, rewarded by what it does to the finishing place
    (cd harness && uv run python price_spots.py --replays ../arena_data --out runs/price_spots.jsonl)
    uv run python opsd/build_price_tables.py --replays arena_data --spots harness/runs/price_spots.jsonl
    uv run python rl/make_puzzle_dataset.py --spots harness/runs/price_spots.jsonl --replays arena_data --prefix pokerbots-rated

Writes `<prefix>-train` and `<prefix>-select`, split by match so no game appears in both. Each task carries the
public record of one hand, the seat and decision to ask about, and the label the reward needs: the seat's equity
against the hands that stayed in, the price of continuing, and a risk factor.

`risk` scales the chips that would be lost. While five or six players remain, losing most of a stack means finishing
last or next to last, which the arena's rating punishes far more than it rewards a win (about -1.45 for last against
+0.83 for first), so those chips count for more than their face value.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from ac2.runtime import Message, Task
from ac2.sdk import Client

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "opsd"))
from poker_harness.replay_split import held_out as is_select  # noqa: E402

from vals_tools.tasks import TABLE_RULES  # noqa: E402

RISK_BUST, RISK_EARLY = 1.5, 1.25  # most of the stack with 5-6 players left; any stack-sized call with 5-6 left


def risk(players_left: int, share_of_stack: float) -> float:
    if players_left >= 5:
        return RISK_BUST if share_of_stack >= 0.8 else RISK_EARLY
    return 1.0


def puzzle_task(puzzle: dict, match: dict, label: dict | None = None) -> Task:
    _, hand_number, seat, action_index = puzzle["id"].split(":")
    seat, action_index = int(seat), int(action_index)
    hand = next(h for h in match["hands"] if h["hand_number"] == int(hand_number))
    logged = hand["actions"][action_index]
    decision = sum(a["seat"] == seat for a in hand["actions"][:action_index])
    if label is None:
        stack_before = logged["stack_after"] + (logged["to_call"] if logged["action"] == "call" else 0)
        if logged["action"] in ("bet", "raise"):  # chips put in beyond the call are not known here: use the call alone
            stack_before = max(stack_before, logged["to_call"])
        share = puzzle["to_call"] / max(1, stack_before)
        label = {
            "equity": puzzle["equity"], "to_call": puzzle["to_call"], "pot": puzzle["pot"],
            "risk": risk(puzzle["players_left"], share), "kind": puzzle["kind"],
        }
    record = {
        "hand_number": hand["hand_number"],
        "button_seat": hand["button_seat"],
        "players": [{"seat": p["seat"], "starting_stack": p["starting_stack"]} for p in hand["players"]],
        "actions": [{"seat": a["seat"], "action": a["action"], "amount": a["amount"], "say": a.get("say") or ""} for a in hand["actions"]],
    }
    return Task(
        input=[Message(role="user", content=f"Answer one decision from seat {seat} in hand {hand_number} of a real game.")],
        env_params=TABLE_RULES
        | {
            "seed": int(match["seed"]),
            "seats": 6,
            "policy_seats": [seat],
            "puzzle": {"hand": record, "seat": seat, "decision": decision, "label": label, "id": puzzle["id"]},
        },
    )


def rated_label(spot: dict) -> dict | None:
    """A label scored in rating points: what folding, losing and winning are worth to the seat's finishing place."""
    from price_card import card

    c = card(spot["messages"])
    if c is None:
        return None
    return {
        "equity": spot["equity"], "to_call": spot["to_call"], "pot": spot["pot"], "kind": c["street"],
        "values": {"fold": round(c["fold_value"], 4), "lose": round(c["lose_value"], 4), "win": round(c["win_value"], 4)},
    }


def rated_best(label: dict) -> str:
    v = label["values"]
    return "continue" if label["equity"] * v["win"] + (1 - label["equity"]) * v["lose"] - v["fold"] > 0 else "fold"


def build_rated(args: argparse.Namespace, matches: dict) -> tuple[dict[str, list[Task]], Counter]:
    """Tasks from `harness/price_spots.py`: every sizeable call decision, scored in rating points. The training split
    keeps as many decisions whose best answer is fold as continue, so folding more often earns nothing by itself."""
    stats: Counter = Counter()
    train: dict[str, list[Task]] = {"fold": [], "continue": []}
    select: list[Task] = []
    for line in args.spots.read_text().splitlines():
        spot = json.loads(line)
        match_id = spot["id"].split(":")[0]
        label = rated_label(spot) if match_id in matches else None
        if label is None:
            stats["skipped"] += 1
            continue
        task = puzzle_task(spot, matches[match_id], label)
        best = rated_best(label)
        if is_select(match_id):
            select.append(task)
            stats[f"select: best {best}"] += 1
        else:
            train[best].append(task)
            stats[f"train before balancing: best {best}"] += 1
    rng = random.Random(0)
    size = min(len(train["fold"]), len(train["continue"]))
    balanced = rng.sample(train["fold"], size) + rng.sample(train["continue"], size)
    rng.shuffle(balanced)
    return {f"{args.prefix}-train": balanced, f"{args.prefix}-select": select}, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default="vals-hackathon")
    parser.add_argument("--prefix", default="pokerbots-puzzle")
    parser.add_argument("--puzzles", type=Path, help="from harness/puzzles.py (chip-scored labels)")
    parser.add_argument("--spots", type=Path, help="from harness/price_spots.py: rating-scored labels, balanced training split")
    parser.add_argument("--replays", type=Path, required=True)
    parser.add_argument("--kinds", default="stack_call,river_bet", help="puzzle kinds to include")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    if bool(args.puzzles) == bool(args.spots):
        parser.error("give exactly one of --puzzles and --spots")

    kinds = set(args.kinds.split(","))
    matches = {}
    for path in args.replays.glob("*.json"):
        match = json.loads(path.read_text())
        matches[match["id"]] = match
    built: dict[str, list[Task]] = {f"{args.prefix}-train": [], f"{args.prefix}-select": []}
    stats: Counter = Counter()
    if args.spots:
        built, stats = build_rated(args, matches)
    for line in ([] if args.spots else args.puzzles.read_text().splitlines()):
        puzzle = json.loads(line)
        match_id = puzzle["id"].split(":")[0]
        if puzzle["kind"] not in kinds or match_id not in matches:
            stats["skipped"] += 1
            continue
        task = puzzle_task(puzzle, matches[match_id])
        split = "select" if is_select(match_id) else "train"
        built[f"{args.prefix}-{split}"].append(task)
        stats[f"{split}: best {puzzle['best']}"] += 1
        stats[f"risk {task.env_params['puzzle']['label']['risk']}"] += 1
    for name, tasks in built.items():
        print(name, len(tasks))
    print(json.dumps(dict(sorted(stats.items()))))
    if not args.upload:
        print("dry run: nothing uploaded (pass --upload to create the datasets)")
        return
    client = Client(project=args.project)
    for name, tasks in built.items():
        client.datasets.create(name, description=f"{len(tasks)} poker puzzle tasks from arena replays.", exist_ok=True)
        for start in range(0, len(tasks), 300):
            client.datasets.add_tasks(name, tasks=tasks[start : start + 300])
        print(f"uploaded {name}: {len(tasks)} tasks")


if __name__ == "__main__":
    main()
