from dataclasses import replace

import pytest
from poker_harness.engine.table import DecisionContext, LegalActions, SeatView
from poker_harness.preflop import PreflopAction, opportunities
from poker_harness.preflop_reference import reference_hint, showdown_equities
from poker_harness.prompts import TableRules, decision_prompt, system_prompt


@pytest.fixture
def blind_defense() -> DecisionContext:
    actions = [
        {"seat": seat, "player": f"seat{seat}", "street": "preflop", "action": {"type": "fold"}} for seat in [2, 3, 4]
    ] + [
        {"seat": 5, "player": "seat5", "street": "preflop", "action": {"type": "raise", "amount": 4}},
        {"seat": 0, "player": "seat0", "street": "preflop", "action": {"type": "fold"}},
    ]
    return DecisionContext(
        4,
        30,
        "preflop",
        1,
        1,
        5,
        ["Ad", "9h"],
        [],
        7,
        [
            SeatView(0, "seat0", 199, 1, "folded"),
            SeatView(1, "seat1", 198, 2, ""),
            *(SeatView(seat, f"seat{seat}", 200, 0, "folded") for seat in [2, 3, 4]),
            SeatView(5, "seat5", 196, 4, ""),
        ],
        actions,
        LegalActions(True, False, 2, "raise", 6, 200),
    )


def test_reference_uses_visible_cards_and_correct_call_price(blind_defense) -> None:
    system = system_prompt(TableRules())
    state = decision_prompt(blind_defense)
    hint = reference_hint(system, state)
    assert "2 into 7 chips" in hint and "22.2%" in hint
    assert all(f"{name} " in hint for name in ("tight", "medium", "wide"))
    assert reference_hint(system, "Recap of hand 3.\n\n" + state) == hint
    assert showdown_equities(("9s", "Ac")) == showdown_equities(("Ad", "9h"))


@pytest.mark.parametrize(
    "unsupported",
    ["all_in", "third_all_in", "multiway", "wrong_pot", "limped", "reraised", "not_big_blind"],
)
def test_reference_skips_betting_states_outside_the_comparison(blind_defense, unsupported) -> None:
    ctx = blind_defense
    if unsupported == "all_in":
        ctx.seats[1] = replace(ctx.seats[1], stack=2)
    elif unsupported == "third_all_in":
        ctx.seats[0] = replace(ctx.seats[0], stack=0, status="all-in")
        ctx = replace(ctx, hand_actions=ctx.hand_actions[:-1])
    elif unsupported == "multiway":
        ctx.seats[0] = replace(ctx.seats[0], status="")
    elif unsupported == "wrong_pot":
        ctx = replace(ctx, pot=8)
    elif unsupported == "limped":
        ctx.hand_actions[0]["action"] = {"type": "call"}
    elif unsupported == "reraised":
        ctx.hand_actions[0]["action"] = {"type": "raise", "amount": 4}
    elif unsupported == "not_big_blind":
        ctx = replace(ctx, seat=0)
    assert reference_hint(system_prompt(TableRules()), decision_prompt(ctx)) == ""


def test_fold_spots_respect_action_order_and_heads_up_blinds() -> None:
    def spots(actions):
        return list(opportunities(actions, [1, 4], button=4, big_blind=2))

    assert spots([PreflopAction(4, "raise", 1, 4), PreflopAction(1, "fold", 2)]) == [
        ("unraised", "late_first_in", "button_first_in"),
        ("single_open_le_3bb", "bb_vs_late_small_open"),
    ]
    assert spots([PreflopAction(4, "call", 1), PreflopAction(1, "check", 0)]) == [
        ("unraised", "late_first_in", "button_first_in"),
        (),
    ]
    assert spots([PreflopAction(4, "raise", 1, 4), PreflopAction(1, "raise", 2, 12), PreflopAction(4, "fold", 8)])[
        -1
    ] == ("reraised",)
