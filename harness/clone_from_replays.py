"""Rebuild what one arena entry saw at each of its decisions, from public arena replays, with the action it took.

    uv run python clone_from_replays.py --replays <dir of replay .json> --entry house-professor-pete --out pete.jsonl

A replay has every seat's cards, stacks, actions and talk, and the table's seed. Each hand the entry played is
replayed on the same engine from the same seed, stacks and button, with every other seat repeating its logged
actions and talk. The entry's seat is a real `ModelSeat`, so the messages it builds are the official event prompt
for that spot. A hand is kept only if the replayed cards and actions match the replay exactly.

Each output line is {"messages": [...chat messages...], "action": {...submit_action arguments...}, plus bookkeeping}.
What cannot be recovered: the entry's scratchpad (never published; it is replayed empty) and its reasoning. The
entry's own table talk is dropped from the target, so a model trained on these lines does not copy catchphrases.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path

from poker_harness.completions import table_rules, with_preset
from poker_harness.engine.table import Decision, DecisionContext
from poker_harness.game import play_session
from poker_harness.seat import Completion, ModelSeat

RULES = table_rules(with_preset({"preset": "event"}))
SCRIPTED = {
    "provider": "replay", "latency_seconds": 0.0, "in_tokens": 0, "out_tokens": 0, "reasoning_tokens": 0,
    "cost": 0.0, "reasoning": "", "fallback_used": False, "error": None,
}


class ReplaySeat:
    """Repeats one seat's logged actions and talk for one hand."""

    def __init__(self, name: str, actions: list[dict]) -> None:
        self.name, self.queue, self.ran_out = name, list(actions), False

    async def decide(self, ctx: DecisionContext, history: str) -> Decision:
        if not self.queue:
            self.ran_out = True
            decision = Decision("check" if ctx.legal.can_check else "fold")
            decision.model_response = SCRIPTED | {"model_name": self.name}
            return decision
        logged = self.queue.pop(0)
        kind = logged["action"]
        decision = Decision(kind, logged["amount"] if kind in ("bet", "raise") else None)
        decision.model_response = SCRIPTED | {"model_name": self.name, "say": logged.get("say") or ""}
        return decision


def target_arguments(logged: dict) -> dict:
    kind = logged["action"]
    return {"action": kind, "amount": logged["amount"]} if kind in ("bet", "raise") else {"action": kind}


def replay_players(hand: dict) -> list[ReplaySeat]:
    by_seat = {seat: [] for seat in range(RULES.seats)}
    for action in hand["actions"]:
        by_seat[action["seat"]].append(action)
    return [ReplaySeat(f"seat{seat}", moves) for seat, moves in by_seat.items()]


async def play_replay_hand(match: dict, hand: dict, players: list) -> tuple[dict, str]:
    """Run one public hand and verify the deal, action sequence, board and stacks."""
    stacks = dict.fromkeys(range(RULES.seats), 0) | {p["seat"]: p["starting_stack"] for p in hand["players"]}
    log = await play_session(
        f"clone-{match['id']}", int(match["seed"]), players, RULES,
        initial_stacks=stacks, first_hand=hand["hand_number"], last_hand=hand["hand_number"],
        button_seat=hand["button_seat"],
    )
    played = log["hands"][0]
    dealt = {state["seat"]: state["hole_cards"] for state in played["starting_state"].values()}
    if dealt != {p["seat"]: p["hole_cards"] for p in hand["players"]}:
        return {}, "cards differ"
    ours = [(a["seat"], a["action"]["type"], a["action"].get("amount")) for a in played["actions"]]
    theirs = [(a["seat"], a["action"], a["amount"] if a["action"] in ("bet", "raise") else None) for a in hand["actions"]]
    if ours != theirs:
        return {}, "actions differ"
    if list(played.get("final_board") or []) != list(hand["board"]):
        return {}, "board differs"
    recorded_stacks = dict.fromkeys(played["ending_stacks"], 0)
    recorded_stacks.update({f"seat{seat}": stack for seat, stack in hand["stacks_after"].items()})
    if played["ending_stacks"] != recorded_stacks:
        return {}, "stacks differ"
    return played, ""


async def replay_hand(match: dict, hand: dict, seat: int) -> tuple[list[dict], str]:
    """The (messages, action) examples for `seat` in one hand, and "" or why the hand was dropped."""
    players: list = replay_players(hand)
    mine = players[seat].queue
    if not mine:
        return [], "no decision"
    examples: list[dict] = []

    async def recorder(messages: list) -> Completion:
        logged = mine[len(examples)] if len(examples) < len(mine) else None
        if logged is None:
            return Completion(text="")
        arguments = json.dumps(target_arguments(logged))
        examples.append({"messages": [dict(m) for m in messages], "action": json.loads(arguments), "logged": logged})
        call = {"id": f"call-{match['id']}-{hand['hand_number']}-{len(examples)}", "type": "function",
                "function": {"name": "submit_action", "arguments": arguments}}
        return Completion(text="", tool_arguments=arguments, tool_calls=[call])

    async def no_cleanup(messages: list) -> Completion:
        return Completion(text="")

    name = f"seat{seat}"
    players[seat] = ModelSeat(name, recorder, RULES, model="replay/target", cleanup_complete=no_cleanup)
    played, problem = await play_replay_hand(match, hand, players)
    if problem:
        return [], problem
    invalid = ("fallback", "format_error", "afk", "timed_out", "token_cap", "coerced", "sat_out", "error")
    kept = [e for e in examples if not any(e["logged"].get(flag) for flag in invalid)]
    for e in kept:
        e.update(match=match["id"], kind=match["kind"], hand=hand["hand_number"], hole=played["starting_state"][name]["hole_cards"],
                 street=e["logged"]["street"])
        del e["logged"]
    return kept, ""


async def run(args: argparse.Namespace) -> None:
    kept: list[dict] = []
    stats: Counter = Counter()
    for path in sorted(args.replays.glob("*.json")):
        match = json.loads(path.read_text())
        seat = next((s["seat"] for s in match["seats"] if s["entry"]["id"] == args.entry), None)
        if seat is None or (args.ranked_only and match["kind"] != "ranked"):
            continue
        stats["matches"] += 1
        for hand in match["hands"]:
            if seat not in {p["seat"] for p in hand["players"]}:
                continue
            stats["hands"] += 1
            try:
                examples, problem = await replay_hand(match, hand, seat)
            except Exception as exc:  # noqa: BLE001 - a hand that can't be replayed is dropped and counted
                examples, problem = [], f"error {type(exc).__name__}"
            if problem:
                stats[f"dropped: {problem}"] += 1
            stats["decisions"] += len(examples)
            kept.extend(examples)
    args.out.write_text("".join(json.dumps(e) + "\n" for e in kept))
    actions = Counter(e["action"]["action"] for e in kept)
    print(json.dumps(dict(stats)), "\nactions:", dict(actions), f"\nwrote {len(kept)} examples to {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replays", type=Path, required=True, help="directory of arena replay JSON files")
    parser.add_argument("--entry", default="house-professor-pete", help="entry id as the replays name it")
    parser.add_argument("--ranked-only", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("runs/clone.jsonl"))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
