import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "opsd"))

import open_card
from build_price_tables import held_out_openings, opening_observations, opening_table

STATE = """## Hand 4 of 30
Your seat: 4 (position 4; 0 = small blind)
Street: preflop
Your hole cards: As 5s
Board: -
Pot (including current bets): 3
Table:
  seat 0: seat0 stack 199, bet 1
  seat 1: seat1 stack 198, bet 2
  seat 2: seat2 stack 200, bet 0, folded
  seat 3: seat3 stack 200, bet 0, folded
  seat 4: seat4 (you) stack 200, bet 0
  seat 5: seat5 [button] stack 200, bet 0
Actions this hand:
  preflop: seat2 (seat 2) fold
  preflop: seat3 (seat 3) fold and says: "i would raise with aces"

Legal actions: fold, call 2, raise to 4-200
Call submit_action with your decision."""


def messages(state=STATE):
    return [{"role": "system", "content": "rules"}, {"role": "user", "content": state}]


def test_public_opening_card_excludes_limps_and_existing_raises():
    spot = open_card.parse(messages())
    assert (spot["position"], spot["hand_class"], spot["live"], spot["raise_to"]) == (
        "CO",
        "Axs",
        6,
        5,
    )
    assert (
        open_card.parse(messages(STATE.replace("(seat 2) fold", "(seat 2) call")))
        is None
    )
    assert (
        open_card.parse(messages(STATE.replace("(seat 2) fold", "(seat 2) raise to 6")))
        is None
    )
    assert open_card.parse(messages(STATE.replace(", raise to 4-200", ""))) is None


def test_card_learns_from_normal_raises_and_respects_legal_amounts():
    spot = open_card.parse(messages())
    observations = (
        [{"spot": spot, "action": "raise", "standard": True, "gain": 0.025}] * 12
        + [{"spot": spot, "action": "raise", "standard": False, "gain": -1.40}] * 12
        + [{"spot": spot, "action": "fold", "standard": False, "gain": 0.0}] * 12
    )
    tables = {"open": opening_table(observations)}
    c = open_card.card(messages(), tables)
    assert c["plan"] == "raise" and c["clear"]
    assert open_card.answer(c, {"action": "raise", "amount": 5}) == "raise"
    assert open_card.answer(c, {"action": "raise", "amount": 200}) == "invalid"
    assert open_card.hint(c).startswith(open_card.MARK)

    short = open_card.card(
        messages(
            STATE.replace("stack 200, bet 0\n", "stack 4, bet 0\n", 1).replace(
                "raise to 4-200", "raise to 4-4"
            )
        ),
        tables,
    )
    assert (
        short["plan"] == "fold" and not short["clear"] and open_card.hint(short) == ""
    )


def test_open_observations_measure_eliminations_and_held_out_overlap():
    row = {
        "id": "ranked-game:4:4:2",
        "messages": messages(),
        "stack_before": 200,
        "stack_after": 300,
        "live_after": 5,
        "bust_place": None,
        "took": "raise",
        "raise_to": 5,
    }
    values = {"6|0": [[200, 0.0]], "5|0": [[300, 0.4]]}
    bust = row | {"stack_after": 0, "bust_place": 6}
    rows = opening_observations([row, bust], values)
    assert [r["gain"] for r in rows] == [0.4, -1.45]
    fold = rows[0] | {"action": "fold", "standard": False, "gain": 0.0}
    training = [rows[0]] * 10 + [fold] * 10
    book = opening_table(training)
    test = [rows[0]] * 3 + [fold] * 3
    scored = held_out_openings(test, book)
    assert scored["eligible"] == scored["matched"] == 6
    assert scored["policies"]["open card"]["rating_points_per_1000"] == 400.0
    assert held_out_openings(test[:-1], book)["matched"] == 0
