"""Seeding the scratchpad with a ledger (training only) and grading the after-hand update (`ledger_weight`)."""

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI, Request
from test_official_rules import OFFICIAL, bot, seats, submit, update
from test_server import TASK, play

from poker_harness.game import play_session
from poker_harness.grading import FAILURE_SCORE, REWARD_OFFSET, GradingError, check_grader_params, grade_log
from poker_harness.ledger import from_params, parse, render
from poker_harness.ledger import update as update_ledger
from poker_harness.prompts import TableRules
from poker_harness.server import create_app, starting_scratchpad

STACKS = (155, 275, 0, 218)
SCRATCH = TableRules(
    **{
        **OFFICIAL.__dict__,
        "seats": 4,
        "scratchpad": True,
        "starting_stacks": STACKS,
        "first_hand": 27,
        "play_hands": 1,
    }
)
LEDGER = {
    "hand": 26,
    "total": 30,
    "me": "seat3",
    "seats": {
        "seat0": {"stack": 155, "n": 26, "vp": 9, "pr": 6, "open": [5, 5, 6]},
        "seat1": {"stack": 275, "n": 26, "vp": 14, "pc": 11},
        "seat2": {"stack": 0},
        "seat3": {"stack": 218, "n": 26, "vp": 8, "pr": 7},
    },
}
FOLD = bot(lambda messages: submit("fold"))


async def one_hand(clean, pad: str = render(from_params(LEDGER))) -> dict:
    """seat3 folds its hand with `pad` in its scratchpad, then answers the after-hand turn with `clean`."""
    table = seats(4, SCRATCH, seat3=bot(lambda m: submit("fold"), clean))
    table[3].scratchpad = pad
    return await play_session("g", 5, table, SCRATCH)


def reference(log: dict, pad: str = render(from_params(LEDGER))) -> str:
    return render(update_ledger(parse(pad), log["hands"][0], "seat3", 30))


async def test_a_correct_update_scores_above_no_update_whatever_the_cards_did() -> None:
    probe = await one_hand(None)  # the deal is fixed by the seed, so the right ledger is the same in every run below
    right = reference(probe)
    assert parse(right).hand == 27 and parse(right).seats["seat0"].counts["n"] == 27
    params = check_grader_params({"ledger_weight": 2})

    perfect = grade_log(await one_hand(lambda m, retry: update(right)), ["seat3"], params)
    stale = grade_log(probe, ["seat3"], params)  # never updated: every seat's count is one hand behind
    garbage = grade_log(await one_hand(lambda m, retry: update("seat0 seems tight")), ["seat3"], params)
    plain = grade_log(probe, ["seat3"])

    assert perfect["artifacts"]["ledger_accuracy"] == 1.0 and garbage["artifacts"]["ledger_accuracy"] == 0.0
    assert 0 < stale["artifacts"]["ledger_accuracy"] < 1 and "Ledger" in stale["reasoning"]
    played = plain["score"] - REWARD_OFFSET  # the chips part, the same in all four
    assert perfect["score"] == pytest.approx(REWARD_OFFSET + played * 0.8 + 1.0)
    assert garbage["score"] == pytest.approx(REWARD_OFFSET + played * 0.8 - 1.0)
    assert perfect["score"] - garbage["score"] == pytest.approx(2.0)  # the weight, always in full
    assert garbage["score"] < stale["score"] < perfect["score"] and "ledger_accuracy" not in plain["artifacts"]


async def test_a_blank_scratchpad_is_graded_against_a_ledger_started_from_zero() -> None:
    probe = await one_hand(None, pad="")
    fresh = reference(probe, pad="")
    assert parse(fresh).seats["seat0"].counts["n"] == 1 and parse(fresh).seats["seat2"].stack == 0
    params = check_grader_params({"ledger_weight": 2})
    started = await one_hand(lambda m, retry: update(fresh), pad="")
    assert grade_log(started, ["seat3"], params)["artifacts"]["ledger_accuracy"] == 1.0
    assert grade_log(probe, ["seat3"], params)["artifacts"]["ledger_accuracy"] == 0.0  # still blank


async def test_it_combines_with_the_placement_reward_and_keeps_failures_at_zero() -> None:
    probe = await one_hand(None)
    both = check_grader_params({"reward": "placement", "hands_left": 3, "ledger_weight": 2})
    placed = grade_log(probe, ["seat3"], check_grader_params({"reward": "placement", "hands_left": 3}))
    reply = grade_log(await one_hand(lambda m, retry: update(reference(probe))), ["seat3"], both)
    assert reply["score"] == pytest.approx(REWARD_OFFSET + (placed["score"] - REWARD_OFFSET) * 0.8 + 1.0)
    assert reply["artifacts"]["reward"] == "placement" and reply["artifacts"]["ledger_accuracy"] == 1.0

    broken = await one_hand(lambda m, retry: update(reference(probe)))
    next(a for a in broken["hands"][0]["actions"] if a["player"] == "seat3")["model_response"]["format_error"] = True
    assert grade_log(broken, ["seat3"], both)["score"] == FAILURE_SCORE


async def test_ledger_settings_are_checked() -> None:
    for bad in ({"ledger_weight": 0}, {"ledger_weight": 10}, {"ledger_weight": "2"}, {"ledger_weight": True}):
        with pytest.raises(GradingError):
            check_grader_params(bad)
    no_pad = TableRules(**{**SCRATCH.__dict__, "scratchpad": False})
    log = await play_session("g", 5, seats(4, no_pad), no_pad)
    with pytest.raises(GradingError):  # no scratchpad at the table: nothing to grade
        grade_log(log, ["seat3"], {"ledger_weight": 2})
    with pytest.raises(GradingError):
        grade_log(await one_hand(None), ["seat1", "seat3"], {"ledger_weight": 2})


def test_a_task_can_start_the_scratchpad_as_a_ledger_or_as_plain_notes() -> None:
    assert starting_scratchpad({}) == ""
    assert starting_scratchpad({"starting_scratchpad": "seat0 is tight"}) == "seat0 is tight"
    assert starting_scratchpad({"starting_ledger": LEDGER}) == render(from_params(LEDGER))


def relay_that_reads_its_notes(seen: list[str]) -> FastAPI:
    """A model that folds, and on the after-hand turn sends back the scratchpad it was shown, unchanged."""
    relay = FastAPI()

    @relay.post("/job/trace/v1/chat/completions")
    async def chat(request: Request) -> dict:
        messages = (await request.json())["messages"]
        pad = next((m["content"] for m in messages if "<<<SCRATCHPAD" in str(m.get("content"))), "")
        seen.append(pad)
        if "is over" in str(messages[-1].get("content")):
            contents = pad.split("<<<SCRATCHPAD\n")[1].split("\nSCRATCHPAD>>>")[0]
            name, arguments = "update_scratchpad", {"contents": contents}
        else:
            name, arguments = "submit_action", {"action": "fold"}
        call = {"id": "c1", "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}
        message = {"role": "assistant", "content": "ok", "tool_calls": [call]}
        return {"choices": [{"message": message, "finish_reason": "tool_calls"}], "usage": {}}

    return relay


async def test_a_seeded_task_shows_the_ledger_to_the_policy_seat_only_and_grades_the_update(tmp_path: Path) -> None:
    env = {
        "preset": "event",
        "seed": 9,
        "seats": 4,
        "starting_stacks": list(STACKS),
        "first_hand": 27,
        "play_hands": 1,
        "policy_seats": [3],
        "starting_ledger": LEDGER,
    }
    task = {"input": TASK["input"], "env_params": env, "grader_params": {"ledger_weight": 2}}
    seen: list[str] = []
    app = create_app(relay_transport=httpx.ASGITransport(app=relay_that_reads_its_notes(seen)), runs_dir=tmp_path)
    status, _ = await play(app, task)
    assert status["status"] == "completed"
    seeded = [pad for pad in seen if "LEDGER v1 | after hand 26 of 30 | me seat3" in pad]
    assert seeded and any("(empty)" in pad for pad in seen)  # the other seats still start blank
    assert all("seat1 stack 275 | n26 vp14" in pad for pad in seeded)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://h") as client:
        body = {"rollout_id": "gw_1", "rollout_status": "completed", "task": task}
        headers = {"Idempotency-Key": "gw_1:grading"}
        assert (await client.post("/submit_grading", json=body, headers=headers)).is_success
        for _ in range(200):
            grading = (await client.post("/get_grading_status", json={"rollout_id": "gw_1"})).json()
            if grading["status"] not in ("queued", "running"):
                break
            await asyncio.sleep(0.02)
        reply = (await client.post("/collect_score", json={"rollout_id": "gw_1"})).json()
    # It sent its notes back unchanged: a real ledger, one hand out of date.
    assert grading["status"] == "completed" and 0 < reply["artifacts"]["ledger_accuracy"] < 1
