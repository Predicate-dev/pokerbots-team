"""Puzzle tasks (`puzzle_tasks.py`): a logged hand replayed to one decision, graded from the label."""

from __future__ import annotations

import asyncio
import json

import pytest

from poker_harness import play_session
from poker_harness.completions import table_rules, with_preset
from poker_harness.engine import ScriptedPlayer
from poker_harness.puzzle_tasks import PuzzleError, check_puzzle, grade_puzzle, play_puzzle, puzzle_value
from poker_harness.seat import Completion

EVENT = table_rules(with_preset({"preset": "event"}))


def logged_hand(seed: int = 5) -> tuple[dict, int, int, dict]:
    """A scripted hand as a replay would record it, and one seat's decision facing a bet."""
    players = [ScriptedPlayer(f"seat{i}", "maniac" if i % 2 else "calling_station", seed) for i in range(6)]
    log = asyncio.run(play_session("g", seed, players, EVENT, last_hand=1))
    played = log["hands"][0]
    hand = {
        "hand_number": 1,
        "button_seat": played["button_seat"],
        "players": [{"seat": state["seat"], "starting_stack": 200} for state in played["starting_state"].values()],
        "actions": [
            {"seat": a["seat"], "action": a["action"]["type"], "amount": a["action"].get("amount"), "say": ""}
            for a in played["actions"]
        ],
    }
    own: dict[int, int] = {}
    for a in played["actions"]:
        index = own.get(a["seat"], 0)
        own[a["seat"]] = index + 1
        if a["call_amount"] > 0 and index >= 1:  # a later decision, so earlier own actions are replayed too
            return hand, a["seat"], index, a
    raise AssertionError("no decision facing a bet in this scripted hand")


def reply(action: str) -> Completion:
    arguments = json.dumps({"action": action})
    call = {"id": "c1", "type": "function", "function": {"name": "submit_action", "arguments": arguments}}
    return Completion(text="", tool_arguments=arguments, tool_calls=[call])


def run(answer: Completion, label: dict) -> tuple[dict, list]:
    hand, seat, decision, _ = logged_hand()
    asked: list = []

    async def policy(messages: list) -> Completion:
        asked.append(messages)
        return answer

    puzzle = check_puzzle({"hand": hand, "seat": seat, "decision": decision, "label": label}, 6)
    return asyncio.run(play_puzzle("p", 5, puzzle, EVENT, policy)), asked


def test_the_policy_is_asked_once_with_its_earlier_actions_replayed() -> None:
    hand, seat, decision, original = logged_hand()
    log, asked = run(reply("call"), {"equity": 0.7, "to_call": original["call_amount"], "pot": original["pot_before"], "risk": 1.5})
    assert len(asked) == 1
    messages = asked[0]
    assert messages[0]["role"] == "system" and "## Hand 1 of 30" in messages[2]["content"]
    assert sum(m["role"] == "assistant" for m in messages) == decision  # the seat's own earlier turns
    assert log["puzzle_result"]["answer"] == "continue"


def test_the_reward_compares_continuing_with_folding() -> None:
    label = {"equity": 0.7, "to_call": 40, "pot": 60, "risk": 1.5}
    assert puzzle_value(label) == pytest.approx(0.7 * 60 - 0.3 * 40 * 1.5)
    called, _ = run(reply("call"), label)
    folded, _ = run(reply("fold"), label)
    broken, _ = run(Completion(text="no tool call here"), label)
    assert grade_puzzle(called)["score"] == pytest.approx(6 + (0.7 * 60 - 0.3 * 40 * 1.5) / 10)
    assert grade_puzzle(folded)["score"] == 6.0
    assert grade_puzzle(broken)["score"] == 0.0
    assert grade_puzzle(called)["artifacts"]["correct"] and not grade_puzzle(folded)["artifacts"]["correct"]
    behind = {"equity": 0.2, "to_call": 150, "pot": 160, "risk": 1.5}
    bad_call, _ = run(reply("call"), behind)
    assert grade_puzzle(bad_call)["score"] == 1.0  # clipped: a big call when far behind
    assert grade_puzzle(run(reply("fold"), behind)[0])["artifacts"]["correct"]


def test_a_puzzle_that_does_not_fit_the_hand_is_refused() -> None:
    hand, seat, _, _ = logged_hand()
    with pytest.raises(PuzzleError):
        check_puzzle({"hand": hand, "seat": seat, "decision": 99, "label": {"equity": 0.5, "to_call": 1, "pot": 1}}, 6)
    with pytest.raises(PuzzleError):
        check_puzzle({"hand": hand, "seat": seat, "decision": 0, "label": {"equity": 1.5, "to_call": 1, "pot": 1}}, 6)
