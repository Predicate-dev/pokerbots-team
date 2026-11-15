"""Whole-hand RL datasets from our own arena games, scored on finishing place.

    (cd harness && uv run python cut_snapshots.py ../arena_data/*.json --arena-entry <org id> --out runs/snapshots_arena.jsonl)
    uv run python rl/make_whole_hand_dataset.py --snapshots harness/runs/snapshots_arena.jsonl --replays arena_data \
        --notes-file opsd/notes/call_checklist.txt --field-model vals-hackathon/<deployment>          # dry run; add --upload
    uv run python rl/train.py --prefix pokerbots-whole --from-checkpoint <ckpt> --no-eval --dry-run

Each task is one hand of a real game we played: the same stacks, button and cards, from our seat, played out in full
by the policy. Every decision in the hand is the model's (whether to open, how much to bet, when to stop), and the
reward is the change in its expected finishing place (`grader_params.reward = "placement"`, harness `grading.py`),
so busting with five players left costs what it costs in the arena and a late call with players already out is
priced fairly. This is the task for opening, bet sizing and end-game play; the call puzzles only cover calls.

- Notes: from hand 2 on the seat starts with `--notes-file` in its scratchpad, as a model that keeps a checklist
  will have. Hand 1 starts empty, as in a real game.
- Opponents: preserve the house styles and seats from each source game. Team seats use the field model answering
  without thinking. Use --house-share for a controlled randomized mix instead.
- Hands: training keeps playable hands and samples the rest. Selection keeps every hand in its held-out games.

Writes `<prefix>-train` and `<prefix>-select` using the same match split as the call and opening benchmarks.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from ac2.sdk import Client

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "opsd"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness" / "src"))
from poker_harness.arena_lineups import replay_roster  # noqa: E402
from poker_harness.house_bots import HOUSE_STYLES  # noqa: E402
from poker_harness.replay_split import held_out  # noqa: E402
from price_card import preflop_class  # noqa: E402

from vals_tools.tasks import Opponents, snapshot_datasets  # noqa: E402

PLAYABLE = ("QQ+", "99-JJ", "22-88", "AK", "AQ-AJ", "Ax", "broadway", "suited")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default="vals-hackathon")
    parser.add_argument("--prefix", default="pokerbots-whole")
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--replays", type=Path, required=True, help="the replay folder, to read which cards we held")
    parser.add_argument("--notes-file", type=Path, required=True)
    parser.add_argument("--field-model", required=True, help="deployment for the model opponents, <project>/<name>")
    parser.add_argument("--entry", default="Friends", help="source entry name or org ID")
    parser.add_argument("--house-share", type=float, help="override the source roster with a random house share")
    parser.add_argument("--other-share", type=float, default=0.15, help="share of unplayable hands to keep")
    parser.add_argument("--placement-scale", type=float, default=25.0,
                        help="reward points per unit of finishing-place value. The harness default of 5 leaves most hands "
                        "within 0.05 of the neutral score; 25 gives rewards about the spread of the call puzzles")
    parser.add_argument("--dump", type=Path, help="also write every task's settings to this file, to check them locally")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    if args.house_share is not None and not 0 <= args.house_share <= 1:
        parser.error("--house-share must be between zero and one")
    notes = args.notes_file.read_text().strip()
    held: dict[tuple[str, int], list[str]] = {}
    rosters = {}
    for path in args.replays.glob("*.json"):
        match = json.loads(path.read_text())
        roster = replay_roster(match, args.entry)
        if roster is None:
            continue
        ours = roster.hero
        rosters[roster.match_id] = roster
        for hand in match["hands"]:
            for player in hand["players"]:
                if player["seat"] == ours:
                    held[(match["id"], hand["hand_number"])] = player["hole_cards"]
    spots, stats = [], Counter()
    lineup_by_spot = {}
    for line in args.snapshots.read_text().splitlines():
        spot = json.loads(line)
        cards = held.get((spot["game_id"], spot["first_hand"]))
        if cards is None:
            stats["no cards on record"] += 1
            continue
        kind = preflop_class(cards)
        rng = random.Random(f"{spot['game_id']}:{spot['first_hand']}")
        if not held_out(spot["game_id"]) and kind not in PLAYABLE and rng.random() >= args.other_share:
            stats["unplayable hand left out"] += 1
            continue
        stats[f"hand: {kind}"] += 1
        roster = rosters[spot["game_id"]]
        if spot["policy_seats"] != [roster.hero]:
            raise ValueError(f"{roster.match_id}: replay and snapshot disagree on the policy seat")
        key = (spot["seed"], spot["first_hand"], roster.hero)
        specs = roster.opponent_specs(spot["starting_stacks"])
        if key in lineup_by_spot and lineup_by_spot[key] != specs:
            raise ValueError(f"{roster.match_id}: ambiguous opponent roster for {key}")
        lineup_by_spot[key] = specs
        spots.append(spot)
    opponents = Opponents(model=args.field_model, api="chat", reasoning_effort=None)
    built = snapshot_datasets(args.prefix, spots, opponents, ledger=False, personas=False, select_game=held_out)
    for tasks in built.values():
        for task in tasks:
            env = task.env_params
            rng = random.Random(f"{env['seed']}:{env['first_hand']}:seats")
            key = (env["seed"], env["first_hand"], env["policy_seats"][0])
            specs = lineup_by_spot[key]
            if args.house_share is not None:
                specs = {
                    seat: f"scripted:{rng.choice(HOUSE_STYLES)}" if rng.random() < args.house_share else {"thinking": False}
                    for seat in specs
                }
            env["opponent_seats"] = specs
            if env["first_hand"] > 1:
                env["starting_scratchpad"] = notes
            task.grader_params["placement_scale"] = args.placement_scale
            stats["tasks with notes"] += env["first_hand"] > 1
            stats["scripted opponent seats"] += sum(isinstance(v, str) for v in env["opponent_seats"].values())
            stats["tasks with house opponents"] += any(isinstance(v, str) for v in specs.values())
            stats.update(v for v in specs.values() if isinstance(v, str))
    for name, tasks in built.items():
        print(name, len(tasks))
    print(json.dumps(dict(sorted(stats.items()))))
    if args.dump:
        args.dump.write_text("".join(json.dumps({"dataset": name, "env_params": t.env_params, "grader_params": t.grader_params}) + "\n"
                                     for name, tasks in built.items() for t in tasks))
    if not args.upload:
        print("dry run: nothing uploaded (pass --upload to create the datasets)")
        if training := built[f"{args.prefix}-train"]:
            print(json.dumps(training[0].env_params)[:900])
            print(json.dumps(training[0].grader_params))
        return
    client = Client(project=args.project)
    for name, tasks in built.items():
        client.datasets.create(name, description=f"{len(tasks)} whole hands from our arena games, scored on finishing place.", exist_ok=True)
        for start in range(0, len(tasks), 300):
            client.datasets.add_tasks(name, tasks=tasks[start : start + 300])
        print(f"uploaded {name}: {len(tasks)} tasks")


if __name__ == "__main__":
    main()
