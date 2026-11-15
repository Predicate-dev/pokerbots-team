"""Cut training spots out of finished game logs, for tasks that start mid-game with real stacks and a real ledger.

    uv run python cut_snapshots.py                                  # every game in runs/, to runs/snapshots.jsonl
    uv run python cut_snapshots.py runs/game_*.json --policy seat0  # only your seat's hands
    uv run python cut_snapshots.py --min-hand 20                    # only the last third of each game

Each line of the output is one hand of one game, set up exactly as the game had it: the stacks, the button, the
game's seed (so the same cards are dealt) and the seat ledger a perfect note-taker would have had by then
(`poker_harness/snapshots.py`). Turn the file into datasets from the repo root:

    uv run python rl/make_dataset.py --snapshots harness/runs/snapshots.jsonl --ledger --prefix pokerbots-real --dry-run

`play_local.py` writes full games to runs/. So does every training rollout, one hand at a time; those are skipped
(`--min-log-hands`), since a one-hand log has no earlier hands to take notes on. The ledger's counters only describe
how seats really play if the opponents you train against are the models that played these games.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from poker_harness.snapshots import snapshots


def cut(
    paths: list[Path], game_hands: int = 30, policy: list[str] | None = None, min_hand: int = 1, min_log_hands: int = 5
) -> tuple[list[dict[str, Any]], Counter]:
    """Snapshots from every usable log in `paths`, and counts of what was read, used and skipped (and why)."""
    out: list[dict[str, Any]] = []
    stats: Counter = Counter()
    for path in sorted(paths):
        stats["logs"] += 1
        try:
            log = json.loads(path.read_text())
            if not isinstance(log, dict) or len(log.get("hands") or []) < min_log_hands:
                stats["skipped: too few hands"] += 1
                continue
            log["game_id"] = log.get("game_id") or path.stem  # games are told apart by it when splitting datasets
            spots = snapshots(log, game_hands, policy, min_hand)
        except (ValueError, KeyError, TypeError) as exc:  # not a game log this tool can use
            stats[f"skipped: {type(exc).__name__}"] += 1
            continue
        stats["games"] += 1
        out.extend(spots)
    stats["snapshots"] = len(out)
    return out, stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs", nargs="*", type=Path, help="game logs (default: every .json in runs/)")
    parser.add_argument("--out", type=Path, default=Path("runs/snapshots.jsonl"))
    parser.add_argument("--policy", action="append", help="only this seat's hands, e.g. seat0 (repeatable)")
    parser.add_argument("--min-hand", type=int, default=1, help="skip hands before this one")
    parser.add_argument("--game-hands", type=int, default=30, help="the length the games were meant to run")
    parser.add_argument("--min-log-hands", type=int, default=5, help="skip logs with fewer hands than this")
    args = parser.parse_args(argv)

    paths = args.logs or sorted(Path("runs").glob("*.json"))
    spots, stats = cut(paths, args.game_hands, args.policy, args.min_hand, args.min_log_hands)
    print(json.dumps(dict(stats)))
    if not spots:
        print("no snapshots: play some full games first (play_local.py writes them to runs/)")
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(spot) + "\n" for spot in spots))
    late = sum(spot["hands_left"] <= 6 for spot in spots)
    print(f"wrote {len(spots)} snapshots from {stats['games']} games to {args.out} ({late} in the last 7 hands)")


if __name__ == "__main__":
    main()
