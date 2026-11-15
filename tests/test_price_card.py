import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "opsd"))

import price_card  # noqa: E402
from price_card import card, hint, parse, postflop_class, preflop_class, stack_value  # noqa: E402

STATE = """## Hand 1 of 30
Your seat: 0 (position 0; 0 = small blind)
Street: preflop
Your hole cards: 9c Jd
Board: -
Pot (including current bets): 203
Table:
  seat 0: seat0 (you) stack 199, bet 1
  seat 1: seat1 stack 198, bet 2
  seat 2: seat2 stack 0, bet 200, all-in
  seat 3: seat3 stack 200, bet 0, folded
  seat 4: seat4 stack 200, bet 0, folded
  seat 5: seat5 [button] stack 200, bet 0, folded
Actions this hand:
  preflop: seat2 (seat 2) raise to 200 and says: "all in baby"
  preflop: seat3 (seat 3) fold

Legal actions: fold, call 199
Call submit_action with your decision."""
MESSAGES = [{"role": "system", "content": "rules"}, {"role": "user", "content": "## Your scratchpad"}, {"role": "user", "content": STATE}]
TABLES = {
    "value": {f"{live}|{stage}": [[100, base], [200, base + 0.2], [400, base + 0.6]]
              for live, base in ((2, 0.6), (3, 0.4), (4, 0.1), (5, -0.1), (6, -0.2)) for stage in range(3)},
    "win": {"pre|offsuit": [0.35, 100], "pre|offsuit|shove": [0.37, 50]},
}


def test_parse_reads_the_spot():
    spot = parse(MESSAGES)
    assert spot["hand"] == 1 and spot["street"] == "preflop" and spot["hole"] == ["9c", "Jd"] and spot["board"] == []
    assert (spot["pot"], spot["to_call"], spot["stack"], spot["live"]) == (203, 199, 199, 6)
    assert spot["all_in"] == 1 and spot["all_in_size"] == 200 and spot["committed"] == 1


def test_parse_skips_decisions_without_a_call():
    no_call = STATE.replace("Legal actions: fold, call 199", "Legal actions: check, bet 2-199")
    assert parse(MESSAGES[:2] + [{"role": "user", "content": no_call}]) is None


def test_hand_classes():
    assert preflop_class(["Ah", "Kd"]) == "AK" and preflop_class(["Qs", "Qd"]) == "QQ+" and preflop_class(["9c", "Jd"]) == "offsuit"
    assert preflop_class(["Kh", "Th"]) == "broadway" and preflop_class(["7h", "6h"]) == "suited"
    assert postflop_class(["Ah", "Kd"], ["Ac", "7d", "2s"]) == "top_pair"
    assert postflop_class(["Qh", "Qd"], ["Jc", "7d", "2s"]) == "overpair"
    assert postflop_class(["8h", "8d"], ["Ac", "Ts", "7c", "Kh"]) == "low_pair"
    assert postflop_class(["Kh", "Jh"], ["Ah", "7h", "2s"]) == "draw"
    assert postflop_class(["Kh", "Jh"], ["Ah", "7h", "2s", "3d", "4d"]) == "nothing"
    assert postflop_class(["Ks", "Jh"], ["4d", "Ac", "Jc", "Td", "Jd"]) == "trips"
    assert postflop_class(["9s", "8h"], ["7d", "6c", "5c"]) == "monster"


def test_busting_is_worth_the_last_place_left():
    assert stack_value(6, 1, 0, TABLES["value"]) == price_card.PLACE_POINTS[6]
    assert stack_value(3, 1, 0, TABLES["value"]) == price_card.PLACE_POINTS[3]
    assert stack_value(6, 1, 150, TABLES["value"]) < stack_value(6, 1, 300, TABLES["value"])


def test_whole_stack_call_with_six_left_needs_far_more_than_the_chip_price():
    c = card(MESSAGES, TABLES)
    assert abs(c["price"] - 199 / 402) < 1e-9
    assert c["need"] > c["price"] + 0.15  # losing is last place
    assert c["plan"] == "fold"
    text = hint(c)
    assert "place 6 of 6" in text and "fold" in text and text.startswith("Your read of this decision.")


def test_same_call_with_three_left_needs_less():
    three = STATE.replace("stack 200, bet 0, folded", "stack 0, bet 0, out")
    c = card(MESSAGES[:2] + [{"role": "user", "content": three}], TABLES)
    assert c["live"] == 3 and c["need"] < 0.5
