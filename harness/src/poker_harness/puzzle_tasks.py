"""Puzzle tasks: one real decision per rollout, graded from the cards that were actually out.

A task with `env_params.puzzle` replays a logged hand (an arena replay) up to one decision of the policy seat. Every
other seat repeats its logged actions and talk, and the policy seat's own earlier actions in the hand are replayed
too, so the model is asked exactly the question that seat was asked in the real game. Its answer is the rollout;
nothing after it is played with the model.

    env_params.puzzle = {
        "hand": {"hand_number", "button_seat", "players": [{"seat", "starting_stack"}],
                 "actions": [{"seat", "action", "amount", "say"}]},      # the public record of the hand
        "seat": 3,            # the policy seat
        "decision": 1,        # which of that seat's actions in this hand is the question (0 = its first)
        "label": {"equity", "to_call", "pot", "risk"},                   # from `harness/puzzles.py`
        "notes": "...",       # optional: the seat's scratchpad when the hand starts (default empty)
    }

The reward compares continuing with folding, in chips, from the seat's real equity against the hands that stayed in:

    value(continue) = equity * pot - (1 - equity) * to_call * risk          value(fold) = 0
    score = 6 + clip(value / 10 chips, -5, +5)                              a failed reply scores 0

`risk` is above 1 when losing the chips would cost more than their face value: with five or six players left, going
broke means finishing last or next to last, which the arena's rating punishes far more than it rewards a win. Because
the reward uses equity and not the cards that fell, sixteen samples of the same puzzle are scored on the decision
alone: every sample that continues gets the same score, every sample that folds gets 6.

A label may instead carry `values`: what folding, calling and losing, and calling and winning are each worth to the
seat's finishing place, in rating points (from `opsd/price_card.py`'s stack-value table). The reward is then in
rating points, not chips, so busting early costs what it really costs and a late call with players already out is
priced fairly:

    gain(continue) = equity * values.win + (1 - equity) * values.lose - values.fold
    score = 6 + clip(gain / 0.1 rating points, -5, +5)
"""

from __future__ import annotations

import json
from typing import Any

from .engine.table import Decision, DecisionContext
from .game import play_session
from .prompts import TableRules
from .seat import Complete, Completion, ModelSeat

SCALE_CHIPS = 10.0  # chips of value per reward point
SCALE_RATING = 0.1  # rating points of value per reward point, for labels with `values`
FOLD_SCORE, FAILURE_SCORE = 6.0, 0.0
_SCRIPTED = {
    "provider": "replay", "latency_seconds": 0.0, "in_tokens": 0, "out_tokens": 0, "reasoning_tokens": 0,
    "cost": 0.0, "reasoning": "", "fallback_used": False, "error": None,
}


class PuzzleError(ValueError):
    """The task's `puzzle` can't be played as written."""


class _Phase:
    """Shared by the table: once the question has been answered, everyone else stops repeating the old hand."""

    answered = False


class _ReplaySeat:
    def __init__(self, name: str, actions: list[dict[str, Any]], phase: _Phase) -> None:
        self.name, self.queue, self.phase = name, list(actions), phase

    async def decide(self, ctx: DecisionContext, history: str) -> Decision:
        if self.phase.answered or not self.queue:  # the real hand no longer applies: get out of the way
            decision = Decision("check" if ctx.legal.can_check else "fold")
            decision.model_response = _SCRIPTED | {"model_name": self.name}
            return decision
        logged = self.queue.pop(0)
        kind = logged["action"]
        decision = Decision(kind, logged.get("amount") if kind in ("bet", "raise") else None)
        decision.model_response = _SCRIPTED | {"model_name": self.name, "say": logged.get("say") or ""}
        return decision


def check_puzzle(puzzle: Any, seats: int) -> dict[str, Any]:
    if not isinstance(puzzle, dict) or not {"hand", "seat", "decision", "label"} <= set(puzzle):
        raise PuzzleError("env_params.puzzle needs hand, seat, decision and label")
    hand, label = puzzle["hand"], puzzle["label"]
    seat, decision = int(puzzle["seat"]), int(puzzle["decision"])
    if not 0 <= seat < seats:
        raise PuzzleError(f"env_params.puzzle.seat must be 0-{seats - 1}")
    own = [a for a in hand.get("actions", []) if a["seat"] == seat]
    if not 0 <= decision < len(own):
        raise PuzzleError("env_params.puzzle.decision is not one of the seat's actions in this hand")
    for key in ("equity", "to_call", "pot"):
        if not isinstance(label.get(key), (int, float)):
            raise PuzzleError(f"env_params.puzzle.label.{key} must be a number")
    if not 0 <= label["equity"] <= 1 or label["to_call"] <= 0 or label["pot"] <= 0:
        raise PuzzleError("env_params.puzzle.label is out of range")
    notes = puzzle.get("notes", "")
    if not isinstance(notes, str) or len(notes) > 5_000:
        raise PuzzleError("env_params.puzzle.notes must be text of at most 5000 characters")
    values = label.get("values")
    if values is not None and not (
        isinstance(values, dict) and all(isinstance(values.get(key), (int, float)) for key in ("fold", "lose", "win"))
    ):
        raise PuzzleError("env_params.puzzle.label.values needs fold, lose and win as numbers")
    return puzzle


async def play_puzzle(
    game_id: str, seed: int, puzzle: dict[str, Any], rules: TableRules, policy_complete: Complete,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Replay the hand to the question, ask the policy once, and return the game log with `puzzle_result`."""
    hand, seat, question = puzzle["hand"], int(puzzle["seat"]), int(puzzle["decision"])
    phase = _Phase()
    by_seat: dict[int, list[dict[str, Any]]] = {s: [] for s in range(rules.seats)}
    for action in hand["actions"]:
        by_seat[action["seat"]].append(action)
    asked = 0

    async def complete(messages: list) -> Completion | str:
        nonlocal asked
        index, asked = asked, asked + 1
        if index < question:  # the seat's own earlier actions in this hand, as they were
            logged = by_seat[seat][index]
            kind = logged["action"]
            arguments = json.dumps({"action": kind, "amount": logged["amount"]} if kind in ("bet", "raise") else {"action": kind})
            call = {"id": f"call-replay-{index}", "type": "function", "function": {"name": "submit_action", "arguments": arguments}}
            return Completion(text="", tool_arguments=arguments, tool_calls=[call])
        if index == question:
            try:
                return await policy_complete(messages)
            finally:
                phase.answered = True
        return Completion(text="")  # after the question: check or fold, no model call

    async def no_cleanup(messages: list) -> Completion:
        return Completion(text="")

    players: list[Any] = [
        ModelSeat(f"seat{s}", complete, rules, model="ac2/policy", timeout_seconds=timeout_seconds, cleanup_complete=no_cleanup)
        if s == seat
        else _ReplaySeat(f"seat{s}", by_seat[s], phase)
        for s in range(rules.seats)
    ]
    players[seat].scratchpad = puzzle.get("notes", "")  # what the seat's notes hold when the hand starts
    stacks = dict.fromkeys(range(rules.seats), 0) | {p["seat"]: p["starting_stack"] for p in hand["players"]}
    log = await play_session(
        game_id, seed, players, rules,
        initial_stacks=stacks, first_hand=hand["hand_number"], last_hand=hand["hand_number"],
        button_seat=hand["button_seat"],
    )
    mine = [a for a in log["hands"][0]["actions"] if a["player"] == f"seat{seat}"]
    if len(mine) <= question:
        raise PuzzleError("the replay never reached the question; the hand record does not match this seed")
    reply = mine[question]
    failed = bool(reply["model_response"].get("format_error"))
    kind = reply["action"]["type"]
    answer = "invalid" if failed else "fold" if kind == "fold" else "continue"
    log["puzzle_result"] = {"answer": answer, "action": kind, "label": puzzle["label"]}
    return log


def puzzle_value(label: dict[str, Any]) -> float:
    """Chips continuing is worth over folding, with the chips at risk weighted by `risk`."""
    equity, risk = float(label["equity"]), float(label.get("risk", 1.0))
    return equity * float(label["pot"]) - (1 - equity) * float(label["to_call"]) * risk


def puzzle_gain(label: dict[str, Any]) -> float:
    """Rating points continuing is worth over folding, for a label with `values`."""
    equity, values = float(label["equity"]), label["values"]
    return equity * float(values["win"]) + (1 - equity) * float(values["lose"]) - float(values["fold"])


def grade_puzzle(log: dict[str, Any]) -> dict[str, Any]:
    result = log["puzzle_result"]
    label, answer = result["label"], result["answer"]
    rated = label.get("values") is not None
    value = puzzle_gain(label) if rated else puzzle_value(label)
    scale, unit = (SCALE_RATING, "rating points") if rated else (SCALE_CHIPS, "chips")
    if answer == "invalid":
        score = FAILURE_SCORE
    elif answer == "fold":
        score = FOLD_SCORE
    else:
        score = FOLD_SCORE + max(-5.0, min(5.0, value / scale))
    best = "continue" if value > 0 else "fold"
    detail = (
        f"fold {label['values']['fold']:+.2f}, lose {label['values']['lose']:+.2f}, win {label['values']['win']:+.2f}"
        if rated else f"risk {label.get('risk', 1.0)}"
    )
    artifacts = {
        "reward": "puzzle-rating" if rated else "puzzle", "answer": answer, "best": best, "correct": answer == best,
        "equity": label["equity"], "to_call": label["to_call"], "pot": label["pot"], "risk": label.get("risk", 1.0),
    }
    artifacts["value_rating" if rated else "value_chips"] = value
    return {
        "score": score,
        "reasoning": f"Puzzle: answered {answer}; continuing was worth {value:+.2f} {unit} (equity {label['equity']:.2f}, "
        f"{label['to_call']} to call into {label['pot']}, {detail}), so the best answer was {best}.",
        "artifacts": artifacts,
    }
