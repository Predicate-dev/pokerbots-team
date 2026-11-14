"""Make training-only datasets for the forked harness. Prints what it would build; uploads only with `--upload`.

    uv run python rl/make_training_datasets.py                       # dry run: every family, sizes and a sample task
    uv run python rl/make_training_datasets.py --family endgame      # one family
    uv run python rl/make_training_datasets.py --family endgame --upload

Each family is uploaded as `<prefix>-<family>-train` and `<prefix>-<family>-select`, so `rl/train.py --prefix
<prefix>-<family>` trains on one and runs its in-run eval on the other:

| Family | Task | Reward |
| --- | --- | --- |
| `strong` | One hand at 100 BB where the policy seat holds a playable hand (`--min-chen`), plus a `--weak-fraction` share of weak hands so folding stays in the data | EV in big blinds (the starter's reward) |
| `endgame` | The last 1 to `--hands-left` hands of a 30-hand game from uneven stacks, some seats already out | Finishing place with a chip term (`grader_params.reward = "placement"`) |
| `mixed` | The first `--mixed-hands` hands of a game against scripted players, persona-prompted opponents and the plain opponent model | EV summed over the hands (`ev_unit_bb` = the number of hands) |

These tasks need this repo's harness: the official one doesn't know `first_hand`, `initial_stacks`, `last_hand` or
`opponent_seats`, and the arena never sets them. Judge any checkpoint on the official `pokerbots-eval` and with
`harness/play_local.py`, which play the official rules.

Token budget. In training, `--max-response-len` on `rl/train.py` is a budget for the whole task, not one decision:
every decision's reasoning, every observation and every after-hand cleanup turn. The starter sizes one hand at
24,576. A K-hand task plays K hands and K cleanup turns, so give it about K times that, and `--max-total-len` 8,192
more:

    hands  --max-response-len  --max-total-len
      1          24576              32768
      2          49152              57344
      3          73728              81920
      4          98304             106496

That rule of thumb has not been run on AC2. A task that runs out is played as check or fold for the rest and scores
0, so check the first step's logs for "reply hit the cap" before a long run. Tasks are capped at 4 hands for the same
reason, and because AC2 carries at most 1 MiB per rollout.

Uploading twice to the same dataset name adds the tasks twice (as `rl/make_dataset.py` does); use a new `--prefix`
to rebuild.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vals_tools.tasks import SEATS, Opponents  # noqa: E402
from vals_tools.training_tasks import (  # noqa: E402
    MAX_TASK_HANDS,
    endgame_tasks,
    house_table_tasks,
    mixed_table_tasks,
    strong_hand_tasks,
)

FAMILIES = ("strong", "endgame", "mixed", "house")
# (train tasks, select tasks) per family, and base seeds no official split uses.
DEFAULT_COUNTS = {"strong": (2000, 500), "endgame": (1000, 250), "mixed": (1000, 250), "house": (1500, 300)}
BASE_SEEDS = {"strong": (7_710_001, 7_720_002), "endgame": (7_730_003, 7_740_004), "mixed": (7_750_005, 7_760_006), "house": (7_770_007, 7_780_008)}


def build(args: argparse.Namespace) -> dict[str, list]:
    effort = None if args.opponent_effort == "none" else args.opponent_effort
    opponents = Opponents(args.opponent_model, args.opponent_api, effort)
    families = FAMILIES if args.family == "all" else (args.family,)
    built: dict[str, list] = {}
    for family in families:
        train = DEFAULT_COUNTS[family][0] if args.count is None else args.count
        if args.select_count is not None:
            select = args.select_count
        else:  # a quarter of the train count, as --help says
            select = DEFAULT_COUNTS[family][1] if args.count is None else args.count // 4
        counts = (train, select)
        for split, count, seed in zip(("train", "select"), counts, BASE_SEEDS[family], strict=True):
            if family == "strong":
                tasks = strong_hand_tasks(count, seed, opponents, args.min_chen, args.weak_fraction, args.seats)
            elif family == "endgame":
                tasks = endgame_tasks(count, seed, opponents, args.hands_left, args.chip_weight, args.seats)
            elif family == "house":
                tasks = house_table_tasks(count, seed, opponents, args.house_hands, args.house_placement, args.seats)
            else:
                tasks = mixed_table_tasks(
                    count, seed, opponents, args.mixed_hands, args.scripted_share, args.persona_share, args.seats
                )
            built[f"{args.prefix}-{family}-{split}"] = tasks
    return built


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", default="vals-hackathon")
    p.add_argument("--prefix", default="pokerbots")
    p.add_argument("--family", choices=[*FAMILIES, "all"], default="all")
    p.add_argument("--count", type=int, help="train tasks per family (default: 2000 strong, 1000 endgame, 1000 mixed)")
    p.add_argument("--select-count", type=int, help="select tasks per family (default: a quarter of the train count)")
    p.add_argument("--opponent-model", default="vals-hackathon/ac-vals-model")
    p.add_argument("--opponent-api", choices=["responses", "chat"], default="chat")
    p.add_argument("--opponent-effort", default="none", help="reasoning effort for the opponents, e.g. medium")
    p.add_argument("--seats", type=int, default=SEATS)
    g = p.add_argument_group("strong")
    g.add_argument("--min-chen", type=int, default=6, help="lowest Chen score that counts as playable (default 6)")
    g.add_argument("--weak-fraction", type=float, default=0.15, help="share of tasks kept below it (default 0.15)")
    g = p.add_argument_group("endgame")
    g.add_argument("--hands-left", type=int, default=3, help=f"longest endgame in hands, 1-{MAX_TASK_HANDS} (default 3)")
    g.add_argument("--chip-weight", type=float, help="chip share of the placement reward (default: the grader's 0.25)")
    g = p.add_argument_group("mixed")
    g.add_argument("--mixed-hands", type=int, default=3, help=f"hands per task, 1-{MAX_TASK_HANDS} (default 3)")
    g.add_argument("--scripted-share", type=float, default=0.4, help="share of opponent seats that are scripted")
    g.add_argument("--persona-share", type=float, default=0.4, help="share that are persona-prompted models")
    g = p.add_argument_group("house")
    g.add_argument("--house-hands", type=int, default=1, help=f"hands per task, 1-{MAX_TASK_HANDS} (default 1)")
    g.add_argument("--house-placement", action="store_true", help="grade on finishing place, not EV")
    p.add_argument("--upload", action="store_true", help="create the datasets on AC2 (default: print only)")
    p.add_argument("--dry-run", action="store_true", help="print only; this is the default")
    return p


def main() -> None:
    args = parser().parse_args()
    if args.upload and args.dry_run:
        raise SystemExit("--upload and --dry-run can't be combined")
    if (args.count is not None and args.count < 0) or (args.select_count is not None and args.select_count < 0):
        raise SystemExit("--count and --select-count can't be negative")
    if args.chip_weight is not None and not 0 <= args.chip_weight <= 1:
        raise SystemExit("--chip-weight must be from 0 to 1")
    built = build(args)
    for name, tasks in built.items():
        sample = {"env_params": tasks[0].env_params, "grader_params": tasks[0].grader_params} if tasks else {}
        print(name, len(tasks), json.dumps(sample))
    if not args.upload:
        print("dry run: nothing uploaded (pass --upload to create these datasets)")
        return
    from ac2.sdk import Client

    client = Client(project=args.project)
    for name, tasks in built.items():
        client.datasets.create(name, description=f"{len(tasks)} training-only poker tasks.", exist_ok=True)
        for start in range(0, len(tasks), 500):
            client.datasets.add_tasks(name, tasks=tasks[start : start + 500])
        print(f"{name}: {len(tasks)} tasks")


if __name__ == "__main__":
    main()
