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
import random
from collections import Counter
from pathlib import Path

from poker_harness.completions import table_rules, with_preset
from poker_harness.engine.table import Decision, DecisionContext
from poker_harness.game import play_session
from poker_harness.ledger import GOAL, Ledger, render, update
from poker_harness.persona import RULE
from poker_harness.seat import Completion, ModelSeat

RULES = table_rules(with_preset({"preset": "event"}))
SCRIPTED = {
    "provider": "replay", "latency_seconds": 0.0, "in_tokens": 0, "out_tokens": 0, "reasoning_tokens": 0,
    "cost": 0.0, "reasoning": "", "fallback_used": False, "error": None,
}


# Standing advice kept on the ledger's GOAL line. The scratchpad is the only text a team controls at the event and
# the model rereads it at the start of every hand, so this is where a strategy can sit in plain view from hand 2 on.
ADVICE = (
    "GOAL finish above as many seats as I can. Last place costs the most and a win adds little over second. "
    "While all six are in: no all-in without QQ+ or AK, fold to shoves otherwise. Once two seats are out I cannot "
    "be last: play normal aggressive poker and take chips. Open-raise about 3bb with pairs, broadways, suited aces "
    "and suited connectors, never limp. Do not fold two pair or better to a small bet. Never name my cards in talk."
)


def seat_read(line) -> str:
    """A short read of one seat from its counters, written after `||`. The thresholds follow the three habits that
    stood out in the arena's replays: a seat that shoves light, one that calls everything, one that folds everything."""
    c, n = line.counts, line.counts["n"]
    if n < 4:
        return ""
    if sum(size >= 100 for size in line.opens) >= 2 or (c["pr"] / n >= 0.3 and any(size >= 100 for size in line.opens)):
        return "shoves light: call only with QQ+ AK, also TT JJ AQ"
    if c["vp"] / n <= 0.1 and n >= 6:
        return "folds almost everything: raise its blinds, believe its raises"
    if c["vp"] / n >= 0.35 and c["pr"] <= 1 and c["pc"] >= 3:
        return "calls everything: value bet big, never bluff"
    if c["pr"] / n >= 0.3:
        return "raises often: tighten up against it, reraise only strong"
    return ""


def notes_text(ledger: Ledger) -> str:
    """The seat ledger as scratchpad text: Parth's layout (`ledger.py`) with the advice on the GOAL line and without
    the bet-sizing persona lines, which this training does not use."""
    for name, line in ledger.seats.items():
        line.note = "" if name == ledger.me else seat_read(line)
    lines = [line for line in render(ledger).split("\n") if line != RULE and not line.startswith("MODE ")]
    return "\n".join(ADVICE if line == GOAL else line for line in lines)


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


async def replay_hand(
    match: dict, hand: dict, seat: int, scratchpad: str = "", need_decision: bool = True,
) -> tuple[list[dict], str, list[dict] | None, dict | None]:
    """The (messages, action) examples for `seat` in one hand; "" or why the hand was dropped; the messages of the
    seat's after-hand scratchpad turn; and the replayed hand record. `scratchpad` is what the seat's notes hold when
    the hand starts."""
    seats = RULES.seats
    by_seat = {s: [a for a in hand["actions"] if a["seat"] == s] for s in range(seats)}
    mine = by_seat[seat]
    if not mine and need_decision:
        return [], "no decision", None, None
    examples: list[dict] = []
    cleanup_messages: list[list[dict]] = []

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
        cleanup_messages.append([dict(m) for m in messages])
        return Completion(text="")

    players: list = []
    for s in range(seats):
        if s == seat:
            target = ModelSeat(f"seat{s}", recorder, RULES, model="replay/target", cleanup_complete=no_cleanup)
            target.scratchpad = scratchpad
            players.append(target)
        else:
            players.append(ReplaySeat(f"seat{s}", by_seat[s]))
    stacks = dict.fromkeys(range(seats), 0) | {p["seat"]: p["starting_stack"] for p in hand["players"]}
    log = await play_session(
        f"clone-{match['id']}", int(match["seed"]), players, RULES,
        initial_stacks=stacks, first_hand=hand["hand_number"], last_hand=hand["hand_number"],
        button_seat=hand["button_seat"],
    )
    played = log["hands"][0]
    dealt = {int(name.removeprefix("seat")): state["hole_cards"] for name, state in played["starting_state"].items()}
    if dealt != {p["seat"]: p["hole_cards"] for p in hand["players"]}:
        return [], "cards differ", None, None
    ours = [(int(a["player"].removeprefix("seat")), a["action"]["type"], a["action"].get("amount")) for a in played["actions"]]
    theirs = [(a["seat"], a["action"], a["amount"] if a["action"] in ("bet", "raise") else None) for a in hand["actions"]]
    if [(s, k, v if k in ("bet", "raise") else None) for s, k, v in ours] != theirs:
        return [], "actions differ", None, None
    if list(played.get("final_board") or []) != list(hand["board"]):
        return [], "board differs", None, None
    kept = [e for e in examples if not (e["logged"].get("fallback") or e["logged"].get("format_error") or e["logged"].get("afk"))]
    for e in kept:
        logged = e["logged"]
        e.update(match=match["id"], kind=match["kind"], hand=hand["hand_number"], hole=dealt[seat],
                 street=logged["street"], seat=seat, action_index=hand["actions"].index(logged),
                 to_call=logged["to_call"], pot_before=logged["pot_before"], board=list(logged["board"]),
                 stack_after=logged["stack_after"])
        del e["logged"]
    return kept, "", (cleanup_messages[0] if cleanup_messages else None), played


async def run(args: argparse.Namespace) -> None:
    kept: list[dict] = []
    stats: Counter = Counter()
    for path in sorted(args.replays.glob("*.json")):
        match = json.loads(path.read_text())
        seat = next((s["seat"] for s in match["seats"] if s["entry"]["id"] == args.entry), None)
        if seat is None or (args.ranked_only and match["kind"] != "ranked"):
            continue
        stats["matches"] += 1
        # With --ledger, the cloned seat and a few others each carry a seat ledger through the match: every hand is
        # replayed with the ledger after the previous hand in the scratchpad, and its after-hand turn becomes a
        # training example whose target is the ledger with this hand added.
        watchers = [seat]
        if args.ledger:
            others = [s["seat"] for s in match["seats"] if s["seat"] != seat]
            watchers += random.Random(match["id"]).sample(others, min(args.ledger_seats, len(others)))
        for watcher in watchers:
            prior, pad = None, ""
            for hand in match["hands"]:
                if watcher not in {p["seat"] for p in hand["players"]}:
                    continue
                cloned = watcher == seat
                stats["hands"] += cloned
                try:
                    examples, problem, cleanup, played = await replay_hand(
                        match, hand, watcher, pad, need_decision=not args.ledger
                    )
                except Exception as exc:  # noqa: BLE001 - a hand that can't be replayed is dropped and counted
                    examples, problem, cleanup, played = [], f"error {type(exc).__name__}", None, None
                if problem:
                    stats[f"dropped: {problem}"] += 1
                    if args.ledger:
                        break  # the ledger would miss this hand: stop this seat's pass through the match
                    continue
                if cloned:
                    stats["decisions"] += len(examples)
                    kept.extend(examples)
                if args.ledger and played is not None:
                    prior = update(prior, played, f"seat{watcher}", RULES.total_hands)
                    pad = notes_text(prior)
                    if cleanup:
                        stats["cleanup"] += 1
                        kept.append({"kind": "cleanup", "messages": cleanup, "contents": pad, "match": match["id"],
                                     "hand": hand["hand_number"], "street": "cleanup", "hole": []})
    args.out.write_text("".join(json.dumps(e) + "\n" for e in kept))
    actions = Counter(e["action"]["action"] for e in kept if "action" in e)
    print(json.dumps(dict(stats)), "\nactions:", dict(actions), f"\nwrote {len(kept)} examples to {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replays", type=Path, required=True, help="directory of arena replay JSON files")
    parser.add_argument("--entry", default="house-professor-pete", help="entry id as the replays name it")
    parser.add_argument("--ranked-only", action="store_true")
    parser.add_argument("--ledger", action="store_true",
                        help="carry a seat ledger in the scratchpad and add after-hand examples that update it")
    parser.add_argument("--ledger-seats", type=int, default=1,
                        help="with --ledger: other seats per match that also contribute after-hand examples")
    parser.add_argument("--out", type=Path, default=Path("runs/clone.jsonl"))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
