import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "opsd"))

import price_card
import terminal_calls as terminal

SHORT = """## Hand 5 of 30
Your seat: 0 (position 0; 0 = small blind)
Street: preflop
Your hole cards: Ah Ad
Board: -
Pot (including current bets): 203
Table:
  seat 0: seat0 (you) stack 40, bet 0
  seat 1: seat1 stack 0, bet 200, all-in
  seat 2: seat2 stack 199, bet 1, folded
  seat 3: seat3 stack 198, bet 2, folded
  seat 4: seat4 stack 200, bet 0, folded
  seat 5: seat5 [button] stack 200, bet 0, folded
Actions this hand:
  preflop: seat1 (seat 1) raise to 200

Legal actions: fold, call 40
Call submit_action with your decision."""
START = dict(enumerate((40, 200, 200, 200, 200, 200)))


def test_capped_caller_gets_neither_excess_chips_nor_elimination_credit():
    messages = [{"role": "user", "content": SHORT}]
    spot = terminal.public_call(messages, START, 1, False)
    assert spot is not None
    folded = terminal.finish_stacks(spot, None)
    won = terminal.finish_stacks(spot, 1)
    lost = terminal.finish_stacks(spot, -1)
    tied = terminal.finish_stacks(spot, 0)
    assert [stacks[:2] for stacks in (folded, won, lost, tied)] == [
        (40, 203),
        (83, 160),
        (0, 243),
        (42, 201),
    ]
    assert all(
        sum(stacks) == sum(START.values()) for stacks in (folded, won, lost, tied)
    )
    c = price_card.card(messages)
    assert c["covered_all_ins"] == 0
    assert c["contestable_pot"] == 43
    assert c["price"] == 40 / 83


def test_last_hand_uses_finish_rank_and_terminal_split_is_a_separate_outcome():
    messages = [{"role": "user", "content": SHORT}]
    spot = terminal.public_call(messages, START, 1, False)
    last = replace(spot, hand=30)
    folded, gains = terminal.values(last, price_card.tables()["value"])
    assert folded == price_card.PLACE_POINTS[6]
    assert gains == {-1: 0.0, 0: 0.0, 1: 0.0}

    spot = replace(spot, board=("Ac", "Kc", "Qc", "Jc", "Tc"))
    assert terminal.outcome(spot, ("2d", "3d"), spot.board) == 0
    assert terminal.robust_action((0.1, 0.08, -0.01)) is None
    assert terminal.robust_action((0.1, 0.08, 0.06)) == "call"
