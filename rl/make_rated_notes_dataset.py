"""Rating-scored puzzle datasets where the seat starts with its notes block already in the scratchpad.

    uv run python rl/make_rated_notes_dataset.py --spots harness/runs/price_spots_v2.jsonl --replays arena_data \\
        --notes-file opsd/notes/checklist_and_guide.txt --prefix pokerbots-ratednotes     # dry run; add --upload

The same tasks as `rl/make_puzzle_dataset.py --spots` (every sizeable call decision, rewarded in rating points, the
training split balanced between fold and continue), with one difference: `env_params.puzzle.notes` holds the notes a
model trained to keep a checklist will have in front of it. The opponent guide adds that seat's prior public
betting counts from the source replay. First-hand decisions keep an empty scratchpad, as in a real game.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from ac2.runtime import Task
from ac2.sdk import Client

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness" / "src"))
import make_puzzle_dataset as base  # noqa: E402
from poker_harness.opponent_reads import replay_betting_evidence  # noqa: E402


def add_notes(built: dict[str, list[Task]], matches: dict, notes: str) -> dict[str, int]:
    puzzles = defaultdict(list)
    for tasks in built.values():
        for task in tasks:
            puzzle = task.env_params["puzzle"]
            hand = puzzle["hand"]["hand_number"]
            if hand > 1:
                puzzles[puzzle["id"].split(":", 1)[0]].append(((hand, puzzle["seat"]), puzzle))
    with_evidence = 0
    for match_id, records in puzzles.items():
        evidence = (
            replay_betting_evidence(matches[match_id], (spot for spot, _ in records))
            if "OPPONENT GUIDE" in notes else {}
        )
        for spot, puzzle in records:
            prior = evidence.get(spot, "")
            puzzle["notes"] = "\n\n".join(filter(None, (notes, prior)))[:5000]
            with_evidence += bool(prior and prior in puzzle["notes"])
    return {"tasks with notes": sum(map(len, puzzles.values())), "tasks with opponent notes": with_evidence}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default="vals-hackathon")
    parser.add_argument("--prefix", default="pokerbots-ratednotes")
    parser.add_argument("--spots", type=Path, required=True)
    parser.add_argument("--replays", type=Path, required=True)
    parser.add_argument("--notes-file", type=Path, required=True)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    notes = args.notes_file.read_text().strip()
    matches = {}
    for path in args.replays.glob("*.json"):
        match = json.loads(path.read_text())
        matches[match["id"]] = match
    built, stats = base.build_rated(args, matches)
    notes_stats = add_notes(built, matches, notes)
    for name, tasks in built.items():
        print(name, len(tasks))
    print(json.dumps(dict(sorted(stats.items())) | notes_stats))
    if not args.upload:
        print("dry run: nothing uploaded (pass --upload to create the datasets)")
        return
    client = Client(project=args.project)
    for name, tasks in built.items():
        client.datasets.create(name, description=f"{len(tasks)} rating-scored poker puzzles with notes.", exist_ok=True)
        for start in range(0, len(tasks), 300):
            client.datasets.add_tasks(name, tasks=tasks[start : start + 300])
        print(f"uploaded {name}: {len(tasks)} tasks")


if __name__ == "__main__":
    main()
