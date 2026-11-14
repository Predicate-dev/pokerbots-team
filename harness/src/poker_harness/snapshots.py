"""Cut training spots out of finished games: one hand each, set up exactly as the game had it.

A made-up mid-game spot (`vals_tools/tasks.py`) has invented stacks and an invented ledger whose counters describe
nobody. A snapshot has the real thing: the stacks, button and deal of hand N of a real game, and the seat ledger as
a perfect note-taker would have had it after hands 1 to N-1. Replayed with the game's seed, the hand deals the same
cards to the same seats. If the opponents in training are the models that played the game, the ledger's counters
describe how those seats really play, which is what makes reading the ledger worth something.

A snapshot is a plain dict of task settings (`starting_stacks`, `first_hand`, `button_seat`, `starting_ledger`, ...)
plus `hands_left` for the placement reward. `vals_tools.tasks.snapshot_task` adds the opponents and builds the task.
It needs only the public part of a game log, so it works on logs with or without the private section.
"""

from __future__ import annotations

from typing import Any

from .ledger import Ledger, to_params, update


def snapshots(
    log: dict[str, Any], game_hands: int = 30, policy: list[str] | None = None, min_hand: int = 1
) -> list[dict[str, Any]]:
    """One snapshot per hand from `min_hand` on, for each seat in `policy` (default: every seat) dealt into it.

    `game_hands` is the length the game was meant to run (a game that ended early has fewer hands in its log). The
    first hand's snapshots have no ledger: a game starts with a blank scratchpad."""
    info = log["session_info"]
    names = info["player_names"]
    if names != [f"seat{i}" for i in range(len(names))]:
        raise ValueError(f"snapshots need seats named seat0..seat{len(names) - 1}, not {names}")
    if "seed" not in info:
        raise ValueError("the log has no seed, so its hands can't be dealt again")
    ledgers: dict[str, Ledger | None] = dict.fromkeys(policy or names)
    if set(ledgers) - set(names):
        raise ValueError(f"policy seats {sorted(set(ledgers) - set(names))} are not at this table")
    out = []
    for hand in log["hands"]:
        number, dealt = hand["hand_number"], hand["starting_state"]
        stacks = [dealt.get(name, {}).get("starting_stack", 0) for name in names]
        for name, ledger in ledgers.items():
            if name not in dealt:
                continue
            if number >= min_hand:
                spot = {
                    "game_id": log.get("game_id"),
                    "seed": info["seed"],
                    "seats": len(names),
                    "first_hand": number,
                    "play_hands": 1,
                    "button_seat": hand["button_seat"],
                    "starting_stacks": stacks,
                    "policy_seats": [names.index(name)],
                    "hands_left": game_hands - number,
                }
                out.append(spot | ({"starting_ledger": to_params(ledger)} if ledger else {}))
            ledgers[name] = update(ledger, hand, name, game_hands)
    return out
