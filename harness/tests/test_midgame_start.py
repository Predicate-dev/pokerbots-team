"""Training-only mid-game starts: `starting_stacks`, `first_hand` and `play_hands` drop a seat into the middle of
an event game. The arena never sets them, and a table without them plays exactly as before."""

import pytest
from test_official_rules import OFFICIAL, bot, last_user, seats, shove, submit

from poker_harness.completions import EVENT_PARAMS, rules_output, table_rules, with_preset
from poker_harness.game import play_session
from poker_harness.prompts import TableRules

FOUR = {**OFFICIAL.__dict__, "seats": 4}


def midgame(**changes) -> TableRules:
    return TableRules(**{**FOUR, **changes})


async def test_one_hand_from_the_middle_of_a_game_shows_its_real_number_and_stacks() -> None:
    rules = midgame(starting_stacks=(310, 150, 0, 340), first_hand=27, play_hands=1)
    seen: list[str] = []

    def fold(messages: list[dict]):
        seen.append(last_user(messages))
        return submit("fold")

    log = await play_session("g", 5, seats(4, rules, seat0=bot(fold)), rules)
    (hand,) = log["hands"]
    assert hand["hand_number"] == 27 and "## Hand 27 of 30" in seen[0]
    assert "stack 310" in seen[0] or "stack 309" in seen[0]  # its own stack, less a blind if it posted one
    assert set(hand["starting_state"]) == {"seat0", "seat1", "seat3"}  # the seat at 0 is out and not dealt in
    assert hand["stacks_before"] == {"seat0": 310, "seat1": 150, "seat3": 340}
    assert log["session_info"]["starting_stacks"] == [310, 150, 0, 340]
    assert sum(log["final_stacks"].values()) == 800 and log["final_stacks"]["seat2"] == 0
    assert log["net_chips"] == {n: log["final_stacks"][n] - s for n, s in zip(log["final_stacks"], (310, 150, 0, 340))}


async def test_a_seat_already_out_ranks_below_one_that_busts_in_the_hands_played() -> None:
    rules = midgame(starting_stacks=(40, 0, 40, 400), first_hand=29)
    log = await play_session("g", 11, [*seats(4, rules, **{f"seat{i}": bot(shove) for i in range(4)})], rules)
    places, busted = log["ranking"]["places"], log["eliminated_on_hand"]
    assert busted["seat1"] == 28  # out before the first hand played
    assert log["hands"][0]["hand_number"] == 29 and len(log["hands"]) <= 2
    assert places["seat1"] == max(places.values())
    assert all(places[name] < places["seat1"] for name in ("seat0", "seat2", "seat3"))


async def test_play_hands_stops_early_and_the_button_moves_as_usual() -> None:
    rules = midgame(starting_stacks=(200, 200, 200, 200), first_hand=10, play_hands=3)
    log = await play_session("g", 2, seats(4, rules), rules)
    assert [h["hand_number"] for h in log["hands"]] == [10, 11, 12]
    assert [h["button_seat"] for h in log["hands"]] == [3, 0, 1]

    out_on_button = midgame(starting_stacks=(200, 200, 200, 0), first_hand=10, play_hands=1)
    log = await play_session("g", 2, seats(4, out_on_button), out_on_button)
    assert log["hands"][0]["button_seat"] == 0  # the last seat is out, so the button is on the next live seat


async def test_bad_mid_game_starts_are_refused() -> None:
    for changes in (
        {"starting_stacks": (200, 200, 200)},  # one stack per seat
        {"starting_stacks": (800, 0, 0, 0)},  # nobody to play
        {"starting_stacks": (200, 200, 200, 200), "reset_stacks": True},  # needs carried stacks
        {"first_hand": 31},
        {"first_hand": 0},
    ):
        rules = midgame(**changes)
        with pytest.raises(ValueError):
            await play_session("g", 1, seats(4, rules), rules)


def test_task_settings_reach_the_rules_and_the_event_table_is_unchanged() -> None:
    event = table_rules(with_preset({"preset": "event"}))
    assert (event.starting_stacks, event.first_hand, event.play_hands) == (None, 1, None)
    assert not {"starting_stacks", "first_hand", "play_hands"} & set(rules_output(event))

    task = {"preset": "event", "starting_stacks": [310, 150, 0, 340, 200, 200], "first_hand": 27, "play_hands": 1}
    rules = table_rules(with_preset(task))
    assert rules.total_hands == EVENT_PARAMS["total_hands"] == 30
    assert (rules.starting_stacks, rules.first_hand, rules.play_hands) == ((310, 150, 0, 340, 200, 200), 27, 1)
    assert rules_output(rules)["first_hand"] == 27
