"""The placement reward (`grader_params.reward = "placement"`): a hand is scored on how much it moved the trained
seat's expected finishing place. Tasks without `reward` are scored on chips, as `test_grading.py` checks."""

import asyncio
from pathlib import Path

import httpx
import pytest
from test_official_rules import OFFICIAL, bot, seats, shove, submit
from test_server import TASK, fake_relay, play

from poker_harness.ev import label_ev
from poker_harness.game import play_session
from poker_harness.grading import FAILURE_SCORE, REWARD_OFFSET, GradingError, check_grader_params, grade_log
from poker_harness.placement import DEFAULT_NOISE_BB, place_value
from poker_harness.prompts import TableRules
from poker_harness.server import create_app

NOISE = DEFAULT_NOISE_BB * 2  # chips, at the 1/2 game
FOLD = bot(lambda messages: submit("fold"))


def rules(stacks: tuple[int, ...], first_hand: int) -> TableRules:
    return TableRules(
        **{
            **OFFICIAL.__dict__,
            "seats": len(stacks),
            "starting_stacks": stacks,
            "first_hand": first_hand,
            "play_hands": 1,
        }
    )


async def one_hand(stacks: tuple[int, ...], first_hand: int, **policies) -> dict:
    table = rules(stacks, first_hand)
    return await play_session("g", 5, seats(len(stacks), table, **policies), table)


def placement(hands_left: int, **more) -> dict:
    return check_grader_params({"reward": "placement", "hands_left": hands_left, **more})


async def test_a_hand_is_scored_on_the_change_in_place_value() -> None:
    start = {"seat0": 310, "seat1": 150, "seat2": 0, "seat3": 340}
    log = await one_hand(tuple(start.values()), 27, seat0=FOLD)
    reply = grade_log(log, ["seat0"], placement(3))
    before = place_value(start, "seat0", 4, NOISE)
    after = place_value(log["final_stacks"], "seat0", 3, NOISE)
    assert reply["score"] == pytest.approx(REWARD_OFFSET + 10 * (after - before))
    assert reply["artifacts"]["place_before"] == pytest.approx(before)
    assert reply["artifacts"]["place_after"] == pytest.approx(after)
    assert reply["artifacts"]["reward"] == "placement" and "Place value" in reply["reasoning"]
    # The same log without the option is scored on chips, with no placement fields.
    chips = grade_log(log, ["seat0"])
    lost_bb = (log["final_stacks"]["seat0"] - 310) / 2
    assert chips["score"] == pytest.approx(REWARD_OFFSET + lost_bb) and "reward" not in chips["artifacts"]


async def test_the_same_chips_matter_more_on_the_last_hand_than_early() -> None:
    everyone_folds = {f"seat{i}": FOLD for i in range(4)}  # seat0 is the small blind and gives it up
    early = await one_hand((200, 200, 200, 200), 2, **everyone_folds)
    late = await one_hand((200, 200, 200, 200), 30, **everyone_folds)
    assert early["final_stacks"] == late["final_stacks"] == {"seat0": 199, "seat1": 201, "seat2": 200, "seat3": 200}
    early_score = grade_log(early, ["seat0"], placement(28))["score"]
    late_score = grade_log(late, ["seat0"], placement(0))["score"]
    assert REWARD_OFFSET - 0.1 < early_score < REWARD_OFFSET  # one chip on hand 2 barely moves the finishing place
    assert late_score == pytest.approx(REWARD_OFFSET - 10 * 0.5)  # on the last hand it is the gap to last place


async def test_folding_while_others_lose_chips_can_gain_places_on_the_last_hand() -> None:
    log = await one_hand((200, 200, 200, 200), 30, seat0=FOLD)  # the other three call each other down
    assert log["final_stacks"]["seat0"] == 199
    below = sum(stack < 199 for stack in log["final_stacks"].values())
    reply = grade_log(log, ["seat0"], placement(0))
    assert below >= 1 and reply["score"] == pytest.approx(REWARD_OFFSET + 10 * (below / 3 - 0.5))
    assert grade_log(log, ["seat0"])["score"] < REWARD_OFFSET  # on chips the same fold is a small loss


def bust_log(hero_end: int) -> dict:
    """seat0 starts hand 20 with 300 against 300/300/300 and loses 300 - `hero_end` to seat1."""
    names = [f"seat{i}" for i in range(4)]
    final = dict.fromkeys(names, 300) | {"seat0": hero_end, "seat1": 600 - hero_end}
    return {
        "session_info": {"big_blind": 2, "player_names": names, "starting_stacks": [300] * 4},
        "final_stacks": final,
        "hands": [
            {
                "hand_number": 20,
                "starting_state": {n: {"starting_stack": 300} for n in names},
                "ending_stacks": final,
                "actions": [{"player": "seat0", "action": {"type": "call"}, "model_response": {}}],
            }
        ],
    }


def test_losing_the_last_chips_costs_more_than_the_chips_before_them() -> None:
    scores = [grade_log(bust_log(end), ["seat0"], placement(10))["score"] for end in (300, 200, 100, 2, 0)]
    assert scores == sorted(scores, reverse=True) and scores[0] == REWARD_OFFSET
    assert scores[-1] == pytest.approx(REWARD_OFFSET - 10 * 0.5)  # from an even table to certain last
    assert scores[-2] > scores[-1]  # two chips and a chair still beat being out


async def test_an_all_in_before_the_river_weighs_both_outcomes_by_equity() -> None:
    start = {"seat0": 100, "seat1": 100, "seat2": 400}
    log = await one_hand(tuple(start.values()), 25, seat0=bot(shove), seat1=bot(shove), seat2=FOLD)
    log = label_ev(log, {"seat0"}, variants=1, runouts=64, redraw=False)
    label = log["hands"][0]["ev"]["seat0"]
    assert label["method"] == "allin" and 0 < label["ev_share"] < 1
    reply = grade_log(log, ["seat0"], placement(5))

    dead = 400 - log["final_stacks"]["seat2"]  # the blind the folder left in the pot, if it posted one
    wins = place_value({"seat0": 200 + dead, "seat1": 0, "seat2": 400 - dead}, "seat0", 5, NOISE, start)
    loses = place_value({"seat0": 0, "seat1": 200 + dead, "seat2": 400 - dead}, "seat0", 5, NOISE, start)
    expected = label["ev_share"] * wins + (1 - label["ev_share"]) * loses
    assert loses == 0.0 and reply["artifacts"]["place_after"] == pytest.approx(expected)
    realized = place_value(log["final_stacks"], "seat0", 5, NOISE, start)
    assert reply["artifacts"]["place_after"] != pytest.approx(realized)  # not just the cards that came


def test_a_format_error_still_scores_zero() -> None:
    log = bust_log(600)
    log["final_stacks"]["seat1"] = 0
    log["hands"][0]["actions"][0]["model_response"]["format_error"] = True
    assert grade_log(log, ["seat0"], placement(10))["score"] == FAILURE_SCORE


def test_placement_settings_are_checked() -> None:
    assert check_grader_params({"reward": "chips"}) == {"reward": "chips"}
    for bad in (
        {"reward": "rank"},
        {"reward": "placement", "hands_left": -1},
        {"reward": "placement", "hands_left": 2.5},
        {"reward": "placement", "hands_left": 3, "placement_scale": 0},
        {"reward": "placement", "hands_left": 3, "placement_noise_bb": "wide"},
        {"hands_left": 3},  # placement settings without the placement reward
    ):
        with pytest.raises(GradingError):
            check_grader_params(bad)
    with pytest.raises(GradingError):  # one seat's finishing place, not a team's
        grade_log(bust_log(200), ["seat0", "seat1"], placement(10))


async def test_a_mid_game_task_plays_and_grades_through_the_server(tmp_path: Path) -> None:
    """The task `vals_tools.tasks.midgame_task` builds, end to end: /submit, then the Grader API."""
    env = {
        "preset": "event",
        "seed": 9,
        "seats": 6,
        "starting_stacks": [155, 275, 0, 218, 190, 362],
        "first_hand": 26,
        "play_hands": 1,
        "policy_seats": [3],
        "ev_labels": True,
    }
    task = {"input": TASK["input"], "env_params": env, "grader_params": {"reward": "placement", "hands_left": 4}}
    requests: list[dict] = []
    app = create_app(relay_transport=httpx.ASGITransport(app=fake_relay(requests)), runs_dir=tmp_path)
    status, outputs = await play(app, task)
    assert status["status"] == "completed"
    (hand,) = outputs["log"]["hands"]
    assert hand["hand_number"] == 26 and "seat2" not in hand["starting_state"]
    first_prompt = next(m["content"] for m in requests[0]["body"]["messages"] if "## Hand" in str(m.get("content")))
    assert "## Hand 26 of 30" in first_prompt and "stack 362" in first_prompt

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://h") as client:
        body = {"rollout_id": "gw_1", "rollout_status": "completed", "task": task}
        assert (await client.post("/submit_grading", json=body, headers={"Idempotency-Key": "gw_1:grading"})).is_success
        for _ in range(200):
            grading = (await client.post("/get_grading_status", json={"rollout_id": "gw_1"})).json()
            if grading["status"] not in ("queued", "running"):
                break
            await asyncio.sleep(0.02)
        reply = (await client.post("/collect_score", json={"rollout_id": "gw_1"})).json()
    assert grading["status"] == "completed"
    artifacts = reply["artifacts"]
    assert artifacts["reward"] == "placement" and 0 < artifacts["place_before"] < 1
    assert reply["score"] == pytest.approx(REWARD_OFFSET + 10 * (artifacts["place_after"] - artifacts["place_before"]))
    assert FAILURE_SCORE < reply["score"] <= 11
