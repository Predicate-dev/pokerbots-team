"""The two personas: the private coin, the MODE line on the ledger, the sizes each mode allows, and their grading."""

import re

import pytest
from test_ledger_reward import LEDGER, SCRATCH
from test_official_rules import bot, last_user, seats, submit, update

from poker_harness.game import play_session
from poker_harness.grading import REWARD_OFFSET, GradingError, check_grader_params, grade_log
from poker_harness.ledger import from_params, parse, render, to_params
from poker_harness.ledger import update as update_ledger
from poker_harness.persona import BIG, RULE, SMALL, block_end, mode_for, next_mode, sizing_checks
from poker_harness.prompts import TableRules


def test_the_private_coin_and_the_ten_hand_blocks() -> None:
    assert [mode_for(card) for card in ("Ah", "7d", "Tc", "2s")] == [SMALL, SMALL, BIG, BIG]
    assert [block_end(hand) for hand in (1, 9, 10, 11, 20, 29)] == [10, 10, 20, 20, 30, 30]
    assert next_mode(None, 0, 1, ["Kc", "Kh"]) == (BIG, 10)  # no line yet: drawn from the first card
    assert next_mode(BIG, 10, 4, ["Ah", "Ad"]) == (BIG, 10)  # held through its block whatever the cards
    assert next_mode(BIG, 10, 10, ["Ah", "2c"]) == (SMALL, 20)  # its last hand is done: a new draw
    assert next_mode("loud", 30, 4, ["9s", "9h"]) == (BIG, 10)  # an unknown mode is replaced
    assert next_mode(SMALL, 10, 10, []) == (SMALL, 10)  # a log that hides the cards changes nothing


def test_the_mode_line_round_trips_and_the_ledger_explains_it() -> None:
    ledger = from_params(LEDGER | {"mode": {"name": "big", "through": 30}})
    lines = render(ledger).splitlines()
    assert lines[3] == RULE and lines[4] == "MODE big through hand 30" and "hearts or diamonds = small" in RULE
    assert parse(render(ledger)) == ledger and to_params(ledger)["mode"] == {"name": "big", "through": 30}
    plain = from_params(LEDGER)
    assert plain.mode is None and "mode" not in to_params(plain) and "\nMODE " not in render(plain)
    assert parse(render(plain)) == plain and parse(render(plain)).notes == []  # the MODES rule is not a note
    with pytest.raises(ValueError):
        from_params(LEDGER | {"mode": {"name": "wild", "through": 30}})


def hero(open_to: int, bet_share: float):
    """Opens to `open_to` before the flop, then bets `bet_share` of the pot whenever it can."""

    def act(messages: list[dict]):
        text = last_user(messages)
        legal = text.split("Legal actions:")[-1]
        if "Street: preflop" in text:
            return submit("raise", open_to) if "raise to" in legal else submit("call")
        pot = int(re.search(r"Pot \(including current bets\): (\d+)", text)[1])
        can_bet = re.search(r"bet to (\d+)-(\d+)", legal)
        if not can_bet:
            return submit("call")
        return submit("bet", min(int(can_bet[2]), max(int(can_bet[1]), round(pot * bet_share))))

    return act


async def hand_with(open_to: int, bet_share: float, clean=None, mode: dict | None = None, rules: TableRules = SCRATCH):
    pad = render(from_params(LEDGER | ({"mode": mode} if mode else {})))
    table = seats(4, rules, seat3=bot(hero(open_to, bet_share), clean))
    table[3].scratchpad = pad
    return await play_session("g", 5, table, rules), pad


async def test_sizes_are_checked_against_the_mode() -> None:
    log, _ = await hand_with(5, 0.4)
    hand = log["hands"][0]
    mine = [(a["street"], a["action"]) for a in hand["actions"] if a["player"] == "seat3"]
    assert mine[0] == ("preflop", {"type": "raise", "amount": 5}) and sum(a["type"] == "bet" for _, a in mine) >= 2
    assert sizing_checks(hand, "seat3", SMALL, 2) == [True] * len(mine)  # 2.5 BB, then 40% of the pot each street
    assert sizing_checks(hand, "seat3", BIG, 2) == [False] * len(mine)
    big, _ = await hand_with(7, 0.75)
    assert all(sizing_checks(big["hands"][0], "seat3", BIG, 2)) and not any(
        sizing_checks(big["hands"][0], "seat3", SMALL, 2)
    )
    assert sizing_checks(hand, "seat0", SMALL, 2) == []  # a seat that only called has nothing to check


def sized(amount: int, street: str = "preflop", kind: str = "raise", pot: int = 3, stack_after: int = 100) -> dict:
    move = {"type": kind} | ({"amount": amount} if kind in ("bet", "raise") else {})
    return {"player": "me", "street": street, "action": move, "pot_before": pot, "stack_after": stack_after}


def other(kind: str, amount: int | None = None) -> dict:
    return sized(amount or 0, kind=kind) | {"player": "them"}


def test_callers_all_ins_and_reraises() -> None:
    def checks(actions: list[dict], mode: str) -> list[bool]:
        return sizing_checks({"actions": actions}, "me", mode, 2)

    assert checks([sized(5)], SMALL) == [True] and checks([sized(6)], SMALL) == [False]
    assert checks([sized(6)], BIG) == [True] and checks([sized(5)], BIG) == [False]
    limped = [other("call"), other("call")]  # each caller before us adds 1 BB to the line
    assert checks([*limped, sized(9)], SMALL) == [True] and checks([*limped, sized(9)], BIG) == [False]
    assert checks([*limped, sized(10)], BIG) == [True]
    assert checks([other("raise", 6), sized(18)], BIG) == []  # a reraise is not an open: no rule
    assert checks([sized(40, stack_after=0)], SMALL) == []  # an all-in is left alone
    assert checks([sized(5, "flop", "bet", pot=10)], SMALL) == [True]
    assert checks([sized(6, "flop", "bet", pot=10)], BIG) == [True] and checks(
        [sized(6, "flop", "bet", pot=10)], SMALL
    ) == [False]
    assert checks([sized(30, "turn", "raise", pot=20)], SMALL) == []  # a raise after the flop: no rule


async def test_the_reference_update_keeps_the_mode_for_its_block_then_draws_a_new_one() -> None:
    log, pad = await hand_with(5, 0.4, mode={"name": "small", "through": 30})
    hand = log["hands"][0]
    kept = update_ledger(parse(pad), hand, "seat3", 30)
    assert (kept.mode, kept.mode_through) == (SMALL, 30)  # hand 27 of a block that runs through 30

    expired = update_ledger(from_params(LEDGER | {"mode": {"name": "small", "through": 27}}), hand, "seat3", 30)
    drawn = mode_for(hand["starting_state"]["seat3"]["hole_cards"][0])
    assert (expired.mode, expired.mode_through) == (drawn, 30)  # its last hand was this one: a new draw, through 30
    fresh = update_ledger(None, hand, "seat3", 30)
    assert (fresh.mode, fresh.mode_through) == (drawn, 30)  # a blank scratchpad gets its first mode the same way


async def test_playing_the_part_is_worth_the_persona_weight_whatever_the_cards_did() -> None:
    small = {"name": "small", "through": 30}
    params = check_grader_params({"persona_weight": 2})
    probe, pad = await hand_with(5, 0.4, mode=small)
    right = render(update_ledger(parse(pad), probe["hands"][0], "seat3", 30))

    def keep(messages: list[dict], retry: bool):
        return update(right)

    in_character = grade_log((await hand_with(5, 0.4, keep, small))[0], ["seat3"], params)
    out_of_character = grade_log((await hand_with(7, 0.75, keep, small))[0], ["seat3"], params)
    forgot_the_line = grade_log(probe, ["seat3"], params)  # sized right, but never wrote the ledger back
    wrong_line = right.replace("MODE small through hand 30", "MODE big through hand 30")
    switched = grade_log((await hand_with(5, 0.4, lambda m, r: update(wrong_line), small))[0], ["seat3"], params)

    assert in_character["artifacts"]["persona"] == 1.0 and "Persona 100%" in in_character["reasoning"]
    assert out_of_character["artifacts"]["persona"] == 0.5  # the line is right, the sizes are not
    assert forgot_the_line["artifacts"]["persona"] == 1.0  # the seeded line is still there and still right
    assert switched["artifacts"]["persona"] == 0.5 and "ledger_accuracy" not in switched["artifacts"]
    chips = grade_log(probe, ["seat3"])["score"] - REWARD_OFFSET
    assert in_character["score"] == pytest.approx(REWARD_OFFSET + chips * 0.8 + 1.0)
    assert switched["score"] == pytest.approx(REWARD_OFFSET + chips * 0.8)


async def test_without_a_mode_in_force_only_the_line_is_graded_and_weights_share_the_band() -> None:
    probe, pad = await hand_with(7, 0.9)  # no MODE line in the seeded ledger: any size is fine this hand
    right = render(update_ledger(parse(pad), probe["hands"][0], "seat3", 30))
    assert "\nMODE " in right and "\nMODE " not in pad
    both = check_grader_params({"ledger_weight": 2, "persona_weight": 2, "reward": "placement", "hands_left": 3})
    wrote_it = grade_log((await hand_with(7, 0.9, lambda m, r: update(right)))[0], ["seat3"], both)
    did_not = grade_log(probe, ["seat3"], both)
    assert wrote_it["artifacts"]["persona"] == 1.0 and wrote_it["artifacts"]["ledger_accuracy"] == 1.0
    assert did_not["artifacts"]["persona"] == 0.0 and 0 < did_not["artifacts"]["ledger_accuracy"] < 1
    placed = grade_log(probe, ["seat3"], check_grader_params({"reward": "placement", "hands_left": 3}))["score"]
    assert wrote_it["score"] == pytest.approx(REWARD_OFFSET + (placed - REWARD_OFFSET) * 0.6 + 1.0 + 1.0)
    assert 1.0 <= did_not["score"] < wrote_it["score"] <= 11.0

    for bad in ({"persona_weight": 0}, {"persona_weight": "2"}, {"ledger_weight": 5, "persona_weight": 5}):
        with pytest.raises(GradingError):
            check_grader_params(bad)
