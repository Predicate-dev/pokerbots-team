"""Training-only setups (`training.py`, the placement reward): opt-in, validated, and absent from official games."""

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI, Request
from test_grading import grade, grading_body
from test_server import fake_relay, play

from poker_harness import TableRules, play_session
from poker_harness.completions import table_rules, with_preset
from poker_harness.engine import ScriptedPlayer
from poker_harness.game import placements
from poker_harness.grading import (
    DEFAULT_CHIP_WEIGHT,
    FAILURE_SCORE,
    REWARD_OFFSET,
    GradingError,
    check_grader_params,
    grade_log,
    hand_reward,
    placement,
    seat_result,
)
from poker_harness.seat import ModelSeat
from poker_harness.server import create_app
from poker_harness.spots import hole_cards
from poker_harness.training import TableStart, TrainingSetupError, opponent_specs, table_start

EVENT = table_rules(with_preset({"preset": "event"}))


def scripted(archetype: str = "calling_station", seats: int = 6, seed: int = 7) -> list[ScriptedPlayer]:
    return [ScriptedPlayer(f"seat{i}", archetype, seed) for i in range(seats)]


def game(seed: int = 7, archetype: str = "calling_station", **start: object) -> dict:
    return asyncio.run(play_session("g", seed, scripted(archetype, seed=seed), EVENT, **start))


# --- nothing changes without the new keys ---------------------------------------------------------------------


def test_a_task_without_training_keys_gets_the_official_start() -> None:
    params = with_preset({"preset": "event", "seed": 1, "policy_seats": [0], "opponent_model": "p/m"})
    assert table_start(params, EVENT, {0}) == TableStart()
    assert opponent_specs(params, 6, {0}) == {}
    session = game()["session_info"]
    assert not {"first_hand", "last_hand", "game_hands"} & set(session)
    assert session["starting_stacks"] == [200] * 6


def test_explicit_official_values_play_the_same_game() -> None:
    plain = game()
    same = game(initial_stacks=dict.fromkeys(range(6), 200), first_hand=1, last_hand=30, button_seat=5)
    assert plain["hands"] == same["hands"] and plain["final_stacks"] == same["final_stacks"]
    assert plain["ranking"] == same["ranking"] and plain["eliminated_on_hand"] == same["eliminated_on_hand"]


def test_six_max_hole_cards_match_what_the_event_table_deals() -> None:
    dealt = game(seed=4242)["hands"][0]["starting_state"]
    assert all(hole_cards(4242, i, seats=6) == dealt[f"seat{i}"]["hole_cards"] for i in range(6))


# --- mid-game starts ------------------------------------------------------------------------------------------


def test_a_late_window_deals_the_same_cards_as_the_full_game() -> None:
    """Calling stations never bust, so hand 28 of the full game has every seat in and the button where a
    `first_hand: 28` start puts it by default."""
    full = game(seed=99)
    start = table_start({"first_hand": 28}, EVENT, {0})
    late = game(seed=99, first_hand=start.first_hand, button_seat=start.button_seat)
    assert [h["hand_number"] for h in late["hands"]] == [28, 29, 30]
    for hand in late["hands"]:
        official = full["hands"][hand["hand_number"] - 1]
        assert hand["button_seat"] == official["button_seat"]
        assert {n: s["hole_cards"] for n, s in hand["starting_state"].items()} == {
            n: s["hole_cards"] for n, s in official["starting_state"].items()
        }
    assert late["session_info"]["first_hand"] == 28 and late["session_info"]["game_hands"] == 30


def test_initial_stacks_carry_and_a_seat_that_starts_busted_ranks_last() -> None:
    stacks = {0: 30, 1: 500, 2: 0, 3: 370, 4: 0, 5: 300}
    log = game(seed=5, archetype="maniac", initial_stacks=stacks, first_hand=29, button_seat=1)
    first = log["hands"][0]
    assert first["hand_number"] == 29 and first["button_seat"] == 1
    assert set(first["starting_state"]) == {"seat0", "seat1", "seat3", "seat5"}  # busted seats are not dealt in
    assert first["stacks_before"] == {"seat0": 30, "seat1": 500, "seat3": 370, "seat5": 300}
    assert log["session_info"]["starting_stacks"] == [30, 500, 0, 370, 0, 300]
    assert sum(log["final_stacks"].values()) == 1200
    assert log["eliminated_on_hand"]["seat2"] == log["eliminated_on_hand"]["seat4"] == 28
    places = log["ranking"]["places"]
    busted_here = set(log["eliminated_on_hand"]) - {"seat2", "seat4"}
    assert places["seat2"] == places["seat4"] == max(places.values())
    assert all(places[name] < places["seat2"] for name in busted_here)


def test_last_hand_stops_the_game_early() -> None:
    log = game(last_hand=3)
    assert [h["hand_number"] for h in log["hands"]] == [1, 2, 3] and log["complete"]
    assert log["session_info"]["last_hand"] == 3


def test_the_default_button_skips_busted_seats() -> None:
    params = {"initial_stacks": [200, 0, 0, 400, 300, 300], "first_hand": 3}  # hand 3's button would be seat 1
    assert table_start(params, EVENT, {0}).button_seat == 3
    assert table_start({"initial_stacks": [240, 240, 240, 240, 240, 0]}, EVENT, {0}).button_seat == 0  # hand 1: seat 5
    assert table_start({"initial_stacks": [200] * 6}, EVENT, {0}).button_seat == 5  # the official hand-1 button


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"initial_stacks": [200, 200]}, "one stack per seat"),
        ({"initial_stacks": [200, -5, 200, 200, 200, 200]}, "negative"),
        ({"initial_stacks": [1200, 0, 0, 0, 0, 0]}, "at least two seats"),
        ({"initial_stacks": [0, 300, 300, 200, 200, 200]}, "policy seats [0]"),
        ({"initial_stacks": [200.5, 200, 200, 200, 200, 200]}, "must be an integer"),
        ({"first_hand": 0}, "first_hand must be 1-30"),
        ({"first_hand": 31}, "first_hand must be 1-30"),
        ({"first_hand": 29, "last_hand": 28}, "last_hand must be 29-30"),
        ({"button_seat": 6}, "outside 0-5"),
        ({"button_seat": 2, "initial_stacks": [200, 400, 0, 200, 200, 200]}, "button_seat 2 has no chips"),
        ({"initial_scratchpad": "x" * 5001}, "limit is 5000"),
        ({"initial_scratchpad": 7}, "string or {seat: string}"),
        ({"initial_scratchpad": {"9": "note"}}, "outside 0-5"),
    ],
)
def test_bad_start_values_are_refused_by_name(params: dict, message: str) -> None:
    with pytest.raises(TrainingSetupError, match="env_params") as error:
        table_start(params, EVENT, {0})
    assert message in str(error.value)


def test_starts_that_need_other_rules_say_so() -> None:
    reset = TableRules(seats=6, reset_stacks=True)
    with pytest.raises(TrainingSetupError, match="carry-over"):
        table_start({"initial_stacks": [200] * 6}, reset, {0})
    with pytest.raises(TrainingSetupError, match="scratchpad table"):
        table_start({"initial_scratchpad": "note"}, TableRules(seats=6), {0})


# --- per-seat opponents ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ({"0": "scripted:nit"}, "policy seat"),
        ({"7": "scripted:nit"}, "outside 0-5"),
        ({"1": "scripted:shark"}, "is not one of"),
        ({"1": "nit"}, "is not one of"),
        ({"1": {"style": "loose"}}, "unknown keys ['style']"),
        ({"1": {"persona": 3}}, "persona must be a string"),
        ({"1": {"api": "grpc"}}, "api must be chat or responses"),
        ({"1": 5}, "a string or an object"),
        (["scripted:nit"], "must be {seat: spec}"),
    ],
)
def test_bad_opponent_specs_are_refused_by_name(spec: object, message: str) -> None:
    with pytest.raises(TrainingSetupError, match="opponent_seats") as error:
        opponent_specs({"opponent_seats": spec, "opponent_model": "p/m"}, 6, {0})
    assert message in str(error.value)


def test_a_model_spec_needs_a_model_from_somewhere() -> None:
    with pytest.raises(TrainingSetupError, match="names no model"):
        opponent_specs({"opponent_seats": {"1": {"persona": "Raise."}}}, 6, {0})
    assert opponent_specs({"opponent_seats": {1: {"model": "p/other"}}}, 6, {0}) == {1: {"model": "p/other"}}


def test_a_persona_is_appended_to_that_seats_system_prompt_only() -> None:
    async def complete(messages: list[dict]) -> str:
        return "{}"

    plain = ModelSeat("seat1", complete, EVENT)
    styled = ModelSeat("seat2", complete, EVENT, persona="Always raise.")
    assert styled.system == f"{plain.system}\n\nAlways raise." and "Always raise." not in plain.system


async def test_one_table_mixes_scripted_persona_and_other_model_seats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AC2_API_KEY", "gateway-key")
    monkeypatch.setenv("POKER_OPPONENT_BASE_URL", "http://relay/job/trace/v1")
    relay_requests: list[dict] = []
    opponent_requests: list[dict] = []
    app = create_app(
        relay_transport=httpx.ASGITransport(app=fake_relay(relay_requests)),
        opponent_transport=httpx.ASGITransport(app=fake_relay(opponent_requests)),
        runs_dir=tmp_path,
    )
    params = {
        "seed": 99,
        "seats": 6,
        "total_hands": 1,
        "policy_seats": [0],
        "opponent_model": "my-project/vanilla",
        "opponent_seats": {
            "1": "scripted:calling_station",
            "2": {"persona": "Always raise."},
            "3": {"model": "my-project/other"},
        },
    }
    status, outputs = await play(app, {"input": [], "env_params": params})
    assert status["status"] == "completed", status
    actions = outputs["log"]["hands"][0]["actions"]
    by_seat = {i: [a for a in actions if a["player"] == f"seat{i}"] for i in range(6)}
    assert all(by_seat[i] for i in range(6))  # everyone calls, so every seat acts

    assert all(a["model_response"]["provider"] == "scripted" for a in by_seat[1])
    assert len(relay_requests) == len(by_seat[0])
    assert len(opponent_requests) == sum(len(by_seat[i]) for i in (2, 3, 4, 5))  # the scripted seat calls no model
    persona = [r for r in opponent_requests if r["body"]["messages"][0]["content"].endswith("\n\nAlways raise.")]
    other = [r for r in opponent_requests if r["body"]["model"] == "my-project/other"]
    assert len(persona) == len(by_seat[2]) and all(r["body"]["model"] == "my-project/vanilla" for r in persona)
    assert len(other) == len(by_seat[3])
    assert not any("Always raise." in r["body"]["messages"][0]["content"] for r in relay_requests + other)
    assert {a["model_response"]["model_name"] for a in by_seat[3]} == {"opponent/my-project/other"}


async def test_a_table_of_scripted_opponents_needs_no_opponent_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("AC2_API_KEY", raising=False)
    monkeypatch.delenv("POKER_OPPONENT_API_KEY", raising=False)
    relay_requests: list[dict] = []
    app = create_app(relay_transport=httpx.ASGITransport(app=fake_relay(relay_requests)), runs_dir=tmp_path)
    styles = ["maniac", "nit", "calling_station", "random", "nit"]
    params = {
        "seed": 3,
        "seats": 6,
        "total_hands": 1,
        "policy_seats": [0],
        "opponent_seats": {str(i + 1): f"scripted:{style}" for i, style in enumerate(styles)},
    }
    status, outputs = await play(app, {"input": [], "env_params": params})
    actions = outputs["log"]["hands"][0]["actions"]
    assert status["status"] == "completed", status
    assert len(relay_requests) == sum(a["player"] == "seat0" for a in actions)


async def test_a_bad_training_key_fails_the_rollout_with_its_name(tmp_path: Path) -> None:
    app = create_app(relay_transport=httpx.ASGITransport(app=fake_relay([])), runs_dir=tmp_path)
    params = {"preset": "event", "seed": 1, "first_hand": 40}
    status, outputs = await play(app, {"input": [], "env_params": params})
    assert status["status"] == "failed" and "env_params.first_hand must be 1-30" in status["error_message"]
    assert outputs["log"] is None


# --- an endgame task, end to end: prompts, scratchpad, placement reward ----------------------------------------


def event_relay(requests: list[dict]) -> FastAPI:
    """Calls (a check when free) on action turns and writes a short note on cleanup turns."""
    relay = FastAPI()

    @relay.post("/job/trace/v1/chat/completions")
    async def chat(request: Request) -> dict:
        body = await request.json()
        requests.append(body)
        last_user = next(m["content"] for m in reversed(body["messages"]) if m["role"] == "user")
        if "is over" in last_user:
            name, arguments = "update_scratchpad", {"contents": f"note {len(requests)}"}
        else:
            name, arguments = "submit_action", {"action": "call"}
        call = {"id": f"c{len(requests)}", "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}
        return {"choices": [{"message": {"role": "assistant", "content": "ok", "tool_calls": [call]}}], "usage": {}}

    return relay


ENDGAME = {
    "preset": "event",
    "seed": 21,
    "policy_seats": [0],
    "first_hand": 29,
    "initial_stacks": [60, 500, 0, 340, 0, 300],
    "initial_scratchpad": "seat3 bluffs rivers",
    "opponent_seats": {"1": "scripted:calling_station", "3": "scripted:nit", "5": "scripted:calling_station"},
}


async def test_an_endgame_task_shows_the_true_hand_and_scores_the_finishing_place(tmp_path: Path) -> None:
    requests: list[dict] = []
    app = create_app(relay_transport=httpx.ASGITransport(app=event_relay(requests)), runs_dir=tmp_path)
    task = {"input": [], "env_params": ENDGAME}
    status, outputs = await play(app, task)
    assert status == {"status": "completed", "annotations": {"hands_played": 2}}, status
    log = outputs["log"]
    assert [h["hand_number"] for h in log["hands"]] == [29, 30]

    users = [[m["content"] for m in r["messages"] if m["role"] == "user"] for r in requests]
    assert "seat3 bluffs rivers" in users[0][0]  # the first hand opens with the task's scratchpad
    assert users[0][1].startswith("## Hand 29 of 30")
    assert "The game lasts 30 hands" in requests[0]["messages"][0]["content"]
    assert any(u[1].startswith("## Hand 30 of 30") and "seat3 bluffs rivers" not in u[0] for u in users if len(u) > 1)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://h") as client:
        body = grading_body("gw_1", env_params=ENDGAME, grader_params={"reward": "placement"})
        graded, score = await grade(client, body)
    assert graded["status"] == "completed", graded
    reply = score.json()
    full = json.loads((tmp_path / "gw_1.json").read_text())
    finish = placement(full, ["seat0"])
    expected = 1 + 10 * ((1 - DEFAULT_CHIP_WEIGHT) * (6 - finish.place) / 5 + DEFAULT_CHIP_WEIGHT * finish.chip_share)
    assert reply["score"] == pytest.approx(expected) and 1.0 <= reply["score"] <= 11.0
    assert reply["artifacts"]["reward"] == "placement" and reply["artifacts"]["hands"] == 2
    assert reply["artifacts"]["place"] == placements(full)["seat0"] and reply["artifacts"]["seats"] == 6
    assert reply["artifacts"]["final_chips"] == full["final_stacks"]["seat0"] and not reply["artifacts"]["busted"]


# --- the grader's modes ---------------------------------------------------------------------------------------


def finished(final: dict[str, int], busted: dict[str, int] | None = None) -> dict:
    """A minimal finished two-hand log with these final stacks and bust hands."""
    names = list(final)
    hand = {
        "hand_number": 1,
        "starting_state": {n: {"starting_stack": 200} for n in names},
        "stacks_before": dict.fromkeys(names, 200),
        "actions": [{"player": "seat0", "action": {"type": "call"}, "model_response": {}}],
    }
    return {
        "hands": [hand, hand | {"hand_number": 2}],
        "session_info": {"player_names": names, "starting_stacks": [200] * len(names), "big_blind": 2},
        "final_stacks": final,
        "eliminated_on_hand": busted or {},
    }


def test_placement_scores_run_from_1_for_last_and_broke_to_11_for_first_with_everything() -> None:
    six = [f"seat{i}" for i in range(6)]
    winner = finished(dict.fromkeys(six, 0) | {"seat0": 1200}, {n: 2 for n in six[1:]})
    loser = finished({"seat0": 0, "seat1": 400, **dict.fromkeys(six[2:], 200)}, {"seat0": 1})
    params = {"reward": "placement"}
    assert grade_log(winner, ["seat0"], params)["score"] == pytest.approx(11.0)
    assert grade_log(loser, ["seat0"], params)["score"] == pytest.approx(1.0)
    assert grade_log(loser, ["seat0"], params)["artifacts"]["busted"] is True


def test_placement_rewards_rank_first_and_chips_within_a_rank() -> None:
    six = [f"seat{i}" for i in range(6)]
    params = {"reward": "placement"}
    ahead = finished({"seat0": 260, "seat1": 250, "seat2": 240, "seat3": 230, "seat4": 220, "seat5": 0}, {"seat5": 2})
    behind = finished({"seat0": 240, "seat1": 250, "seat2": 260, "seat3": 230, "seat4": 220, "seat5": 0}, {"seat5": 2})
    richer = finished({"seat0": 500, "seat1": 250, "seat2": 100, "seat3": 130, "seat4": 220, "seat5": 0}, {"seat5": 2})
    first, third, first_rich = (grade_log(log, ["seat0"], params)["score"] for log in (ahead, behind, richer))
    assert first > third and first_rich > first  # a better place beats chips; more chips break ties in place
    # Surviving on a short stack still beats busting with nothing.
    short = finished({"seat0": 20, "seat1": 480, **dict.fromkeys(six[2:], 175)}, {})
    bust = finished({"seat0": 0, "seat1": 500, **dict.fromkeys(six[2:], 175)}, {"seat0": 2})
    assert grade_log(short, ["seat0"], params)["score"] == pytest.approx(1 + 10 * 0.25 * 20 / 1200)
    assert grade_log(bust, ["seat0"], params)["score"] == pytest.approx(1.0)
    # With chip_weight 0 only the place counts; with 1 only the chips do.
    assert grade_log(ahead, ["seat0"], params | {"chip_weight": 0})["score"] == pytest.approx(11.0)
    assert grade_log(ahead, ["seat0"], params | {"chip_weight": 1})["score"] == pytest.approx(1 + 10 * 260 / 1200)


def test_a_format_error_scores_zero_in_every_mode() -> None:
    log = finished({"seat0": 1200, **{f"seat{i}": 0 for i in range(1, 6)}}, {f"seat{i}": 2 for i in range(1, 6)})
    log["hands"][0]["actions"][0]["model_response"]["format_error"] = True
    assert grade_log(log, ["seat0"], {"reward": "placement"})["score"] == FAILURE_SCORE
    assert grade_log(log, ["seat0"], {})["score"] == FAILURE_SCORE


def test_the_default_reward_is_unchanged_and_ev_unit_scales_it() -> None:
    log = game(seed=11, archetype="random", last_hand=3)
    assert grade_log(log, ["seat0"]) == grade_log(log, ["seat0"], {}) == grade_log(log, ["seat0"], {"reward": "ev"})
    result = seat_result(log, ["seat0"])
    assert hand_reward(result, {}) == pytest.approx(result.ev_bb)
    assert hand_reward(result, {"ev_unit_bb": 3}) == pytest.approx(result.ev_bb / 3)
    scaled = grade_log(log, ["seat0"], {"ev_unit_bb": 3})["score"]
    assert scaled == pytest.approx(REWARD_OFFSET + max(-5.0, min(5.0, result.ev_bb / 3)))
    assert "place" not in grade_log(log, ["seat0"])["artifacts"]


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"reward": "rank"}, "reward must be one of"),
        ({"reward": "placement", "chip_weight": 1.5}, "chip_weight must be a number from 0 to 1"),
        ({"reward": "placement", "chip_weight": "half"}, "chip_weight must be a number"),
        ({"chip_weight": 0.5}, "only applies to reward 'placement'"),
        ({"reward": "placement", "ev_unit_bb": 3}, "only applies to reward 'ev'"),
        ({"ev_unit_bb": 0}, "ev_unit_bb must be a number"),
        ({"reward": "placement", "bonus": 1}, "unknown grader_params"),
    ],
)
def test_bad_grader_params_fail_the_grading(params: dict, message: str) -> None:
    with pytest.raises(GradingError) as error:
        check_grader_params(params)
    assert message in str(error.value)


def test_good_grader_params_pass() -> None:
    for params in ({}, {"reward": "ev"}, {"reward": "ev", "ev_unit_bb": 3}, {"reward": "placement", "chip_weight": 0.4}):
        assert check_grader_params(params) == params
