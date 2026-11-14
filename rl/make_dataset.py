"""Make the RL datasets: single 6-max hands at 100 BB on fresh seeds, one AC2 task per hand.

Writes <prefix>-train, -select (for picking checkpoints), -eval (held out), -smoke and -table-smoke. Every task
starts from the harness's "event" preset. Runs on these datasets name no grader, so the harness's sample
grader (harness/src/poker_harness/grading.py) scores each hand by its EV in big blinds.

    uv run python rl/make_dataset.py --dry-run
    uv run python rl/make_dataset.py
    uv run python rl/make_dataset.py --placement --prefix pokerbots-place --dry-run   # finishing-place datasets
    uv run python rl/make_dataset.py --placement --ledger --prefix pokerbots-ledger --dry-run   # + seat ledger
    uv run python rl/make_dataset.py --placement --personas --prefix pokerbots-persona --dry-run   # + the personas
    uv run python rl/make_dataset.py --snapshots harness/runs/snapshots.jsonl --ledger --prefix pokerbots-real --dry-run
    uv run python rl/make_dataset.py --opponent-model vals-hackathon/opponent --opponent-api responses --opponent-effort medium

By default the opponents are `vals-hackathon/ac-vals-model`, the model every team org starts with.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ac2.sdk import Client

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vals_tools.tasks import SEATS, Opponents, datasets, placement_datasets, snapshot_datasets  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default="vals-hackathon")
    parser.add_argument("--prefix", default="pokerbots")
    parser.add_argument(
        "--opponent-model",
        default="vals-hackathon/ac-vals-model",
        help="model the opponent seats call (default: the model every team org starts with)",
    )
    parser.add_argument(
        "--opponent-api", choices=["responses", "chat"], default="chat", help="responses for OpenAI reasoning models"
    )
    parser.add_argument("--opponent-effort", default="none", help="reasoning effort for the opponents, e.g. medium")
    parser.add_argument("--seats", type=int, default=SEATS)
    parser.add_argument(
        "--placement",
        action="store_true",
        help="make the finishing-place datasets instead: half mid-game hands scored on placement, half fresh hands "
        "scored on chips. Give them their own --prefix, e.g. pokerbots-place",
    )
    parser.add_argument(
        "--ledger",
        action="store_true",
        help="with --placement or --snapshots: also start scratchpads as a seat ledger and grade how correctly it "
        "is updated",
    )
    parser.add_argument(
        "--personas",
        action="store_true",
        help="with --placement or --snapshots: also grade the two bet-sizing personas (the ledger's MODE line and "
        "sizes that fit it). Brings --ledger with it",
    )
    parser.add_argument(
        "--snapshots",
        type=Path,
        help="make -train and -select from hands cut out of real games (harness/cut_snapshots.py writes this file), "
        "scored on placement. Needs its own --prefix, e.g. pokerbots-real",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    effort = None if args.opponent_effort == "none" else args.opponent_effort
    if (args.placement or args.snapshots) and args.prefix == "pokerbots":
        parser.error("--placement and --snapshots need their own --prefix so they can't overwrite pokerbots-train")
    if args.placement and args.snapshots:
        parser.error("--placement makes up its spots and --snapshots reads real ones: pick one")
    if (args.ledger or args.personas) and not (args.placement or args.snapshots):
        parser.error("--ledger and --personas go with --placement or --snapshots")
    opponents = Opponents(args.opponent_model, args.opponent_api, effort)
    if args.snapshots:
        spots = [json.loads(line) for line in args.snapshots.read_text().splitlines() if line.strip()]
        built = snapshot_datasets(args.prefix, spots, opponents, ledger=args.ledger, personas=args.personas)
        for name in [name for name, tasks in built.items() if not tasks]:
            print(f"{name}: no tasks (too few games to hold one out); train with --no-eval or cut more games")
            del built[name]
    elif args.placement:
        built = placement_datasets(args.prefix, opponents, args.seats, ledger=args.ledger, personas=args.personas)
    else:
        built = datasets(args.prefix, opponents, args.seats)
    if args.dry_run:
        for name, tasks in built.items():
            print(name, len(tasks), tasks[0].env_params)
        return
    client = Client(project=args.project)
    for name, tasks in built.items():
        client.datasets.create(name, description=f"{len(tasks)} {args.seats}-max poker tasks.", exist_ok=True)
        for start in range(0, len(tasks), 500):
            client.datasets.add_tasks(name, tasks=tasks[start : start + 500])
        print(f"{name}: {len(tasks)} tasks")


if __name__ == "__main__":
    main()
