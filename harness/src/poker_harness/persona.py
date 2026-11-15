"""Two personas that play the same hands and differ in how they size bets.

An opponent taking notes writes down what it can see, and sizes are what it sees most: "seat3 opens to 5 and bets a
third of the pot". The two modes below give it two different players to write down:

    small   opens to 2.5 BB or less, bets half the pot or less
    big     opens to 3 BB or more, bets 60% of the pot or more

Both are ordinary, winning ways to play, and neither changes which hands are played or how often, so the stats a
tracker counts (how often we play, raise, fold) are the same in both. One is not a decoy for the other: every hand
is scored on its result in either mode, so both are trained to win.

The mode is held for a block of 10 hands and written on the ledger's MODE line. Each new block's mode comes from a
coin no one else can see: the suit of our first hole card on the hand the block is chosen (hearts or diamonds =
small, clubs or spades = big). A fixed schedule could be learned from replays; this can't be predicted, and half
the time the next block looks like a different player.

Only what has a clear right answer is checked (`sizing_checks`): the size of an open raise before the flop and of a
bet after it. Reraises, raises after the flop and all-ins are left to the model.
"""

from __future__ import annotations

from typing import Any

SMALL, BIG = "small", "big"
MODES = (SMALL, BIG)
BLOCK = 10  # hands a mode is held for
SMALL_OPEN_BB, BIG_OPEN_BB = 2.5, 3.0  # open-raise "to" sizes, plus 1 BB for each seat that called before us
SMALL_BET, BIG_BET = 0.5, 0.6  # postflop bet as a share of the pot
RULE = (
    "MODES same hands either way, only sizes change. small: open to 2.5 BB or less, bet half pot or less. big: open "
    "to 3 BB or more, bet 60% pot or more. Add 1 BB to an open per caller before me. Hold the MODE through its last "
    "hand; when there is none or that hand is done, pick the next from my first hole card this hand: hearts or "
    f"diamonds = small, clubs or spades = big, through the next multiple of {BLOCK}."
)


def mode_for(card: str) -> str:
    """The private coin: the mode a hole card such as "Ah" or "7c" picks."""
    return SMALL if card[-1].lower() in "hd" else BIG


def block_end(hand: int) -> int:
    """The last hand of the block a mode chosen after `hand` runs through: the next multiple of 10 past it."""
    return (hand // BLOCK + 1) * BLOCK


def next_mode(mode: str | None, through: int, hand: int, hole_cards: list[str]) -> tuple[str | None, int]:
    """The MODE line after `hand`: kept while its block lasts, else drawn from this hand's first hole card. Without
    hole cards (a log that hides them) there is nothing to draw from and the line stays as it was."""
    if mode in MODES and hand < through:
        return mode, through
    if not hole_cards:
        return mode, through
    return mode_for(hole_cards[0]), block_end(hand)


def sizing_checks(hand: dict[str, Any], seat: str, mode: str, big_blind: int) -> list[bool]:
    """One verdict for each of `seat`'s sized actions this hand that `mode` has a rule for: its open raise before
    the flop (if it was first to raise) and each bet after it. All-ins are skipped."""
    checks: list[bool] = []
    raised, callers = False, 0
    for action in hand["actions"]:
        kind, amount = action["action"]["type"], action["action"].get("amount")
        preflop = action["street"] == "preflop"
        mine = action["player"] == seat and action["stack_after"] > 0 and amount is not None
        if preflop and kind in ("bet", "raise"):
            if mine and not raised:
                line = (SMALL_OPEN_BB if mode == SMALL else BIG_OPEN_BB) * big_blind + callers * big_blind
                checks.append(amount <= line if mode == SMALL else amount >= line)
            raised = True
        elif preflop and kind == "call" and not raised:
            callers += 1
        elif not preflop and kind == "bet" and mine:
            share = amount / action["pot_before"] if action["pot_before"] else 1.0
            checks.append(share <= SMALL_BET if mode == SMALL else share >= BIG_BET)
    return checks
