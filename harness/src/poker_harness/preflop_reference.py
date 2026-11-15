"""A check-down comparison for a big blind facing a single small preflop open."""

from __future__ import annotations

import random
import re
from functools import lru_cache
from itertools import combinations

from pokerkit import Deck, StandardHighHand

from .preflop import clockwise
from .spots import RANKS, chen_score


# Restrict the calculation to the official first-decision prompt. Its complete state and action list
# let us exclude capped calls, side pots, limps and reopened betting without reading private data.
_DECISION = re.compile(
    r"## Hand \d+ of \d+\nYour seat: (?P<hero>\d+) \(position \d+; 0 = small blind\)\n"
    r"Street: preflop\nYour hole cards: (?P<hole>[2-9TJQKA][cdhs] [2-9TJQKA][cdhs])\n"
    r"Board: -\nPot \(including current bets\): (?P<pot>\d+)\nTable:\n"
    r"(?P<seats>(?:  seat [^\n]+\n)+)Actions this hand:\n(?P<actions>(?:  preflop: [^\n]+\n)+)\n"
    r"Legal actions: fold, call (?P<call>\d+)(?:, raise to \d+-\d+)?\nCall submit_action with your decision\."
)
_SEAT = re.compile(r"  seat (\d+): .+? stack (\d+), bet (\d+)(?:, (folded|all-in|out))?")
_ACTION = re.compile(r'  preflop: .+? \(seat (\d+)\) (?:fold|raise to (\d+))(?: and says: "[^"\n]*")?')
_TRIALS = 1024
_RANGES = (("tight", 8), ("medium", 6), ("wide", 5))


@lru_cache(maxsize=169)
def _showdown_equities(hole: tuple[str, str]) -> tuple[float, ...]:
    cards = {repr(card): card for card in Deck.STANDARD}
    hero = tuple(cards[code] for code in hole)
    deck = [code for code in cards if code not in hole]
    hands = [(pair, chen_score(pair)) for pair in combinations(deck, 2)]
    shares = []
    total_share, total_combos = 0.0, 0
    upper = max(score for _, score in hands) + 1
    for _, minimum in _RANGES:
        rng = random.Random(":".join((*hole, str(minimum))))
        band = [pair for pair, score in hands if minimum <= score < upper]
        share = 0.0
        for _ in range(_TRIALS):
            opponent = rng.choice(band)
            board = tuple(cards[code] for code in rng.sample([code for code in deck if code not in opponent], 5))
            mine = StandardHighHand.from_game(hero, board)
            theirs = StandardHighHand.from_game(tuple(cards[code] for code in opponent), board)
            share += 1.0 if mine > theirs else 0.5 if mine == theirs else 0.0
        # Wider ranges include the earlier bands. Reuse the same worlds and weight by combo count.
        total_share += share / _TRIALS * len(band)
        total_combos += len(band)
        shares.append(total_share / total_combos)
        upper = minimum
    return tuple(shares)


def showdown_equities(hole: tuple[str, str]) -> dict[str, float]:
    """Raw preflop equity vs nested Chen-score ranges, with suit-equivalent hands cached."""
    first, second = sorted(hole, key=lambda code: RANKS.index(code[0]), reverse=True)
    if first == second:
        raise ValueError("hole cards must be distinct")
    equivalent = (first[0] + "c", second[0] + ("c" if first[1] == second[1] else "d"))
    return {name: share for (name, _), share in zip(_RANGES, _showdown_equities(equivalent), strict=True)}


def reference_hint(system: str, user: str) -> str:
    """Return teacher-only price and range evidence if the prompt supports the comparison."""
    blinds = re.search(r"^- Blinds are (\d+)/(\d+)\.", system, re.MULTILINE)
    # The harness may prepend recaps of finished hands to a new hand's prompt.
    _, header, state = user.strip().rpartition("\n\n## Hand ")
    match = _DECISION.fullmatch(f"## Hand {state}" if header else state)
    if not blinds or not match:
        return ""
    small_blind, big_blind = map(int, blinds.groups())
    if not 0 < small_blind < big_blind:
        return ""
    hero, pot, price = (int(match[key]) for key in ("hero", "pot", "call"))
    seats = {}
    button = None
    for line in match["seats"].splitlines():
        seat = _SEAT.fullmatch(line)
        if not seat:
            return ""
        number, stack, bet = map(int, seat.groups()[:3])
        seats[number] = (stack, bet, seat[4])
        if " [button] stack " in line:
            button = number
    dealt = {seat for seat, (_, _, status) in seats.items() if status != "out"}
    if button not in dealt or len(dealt) < 2:
        return ""
    order = clockwise(dealt, button)
    small, big = order[:2] if len(order) > 2 else (button, order[0])
    if hero != big:
        return ""
    raises = []
    for line in match["actions"].splitlines():
        action = _ACTION.fullmatch(line)
        if not action or int(action[1]) == hero:
            return ""
        if action[2] is not None:
            raises.append((int(action[1]), int(action[2])))
    if len(raises) != 1:
        return ""
    opponent, raise_to = raises[0]
    if {seat for seat, (_, _, status) in seats.items() if status not in ("folded", "out")} != {hero, opponent}:
        return ""
    hero_stack, hero_bet, _ = seats[hero]
    opponent_stack, opponent_bet, _ = seats[opponent]
    dead_blind = small_blind if opponent != small else 0
    if (
        not 0 < price < hero_stack
        or opponent_stack <= 0
        or hero_bet != big_blind
        or not big_blind < raise_to <= 3 * big_blind
        or raise_to != opponent_bet
        or price != raise_to - big_blind
        or pot != hero_bet + opponent_bet + dead_blind
    ):
        return ""
    first, second = match["hole"].split()
    if first == second:
        return ""
    equities = showdown_equities((first, second))
    price_share = price / (pot + price)
    comparison = ", ".join(
        f"{name} {share:.0%} equity ({price_share / share:.0%} realization to break even)"
        for name, share in equities.items()
    )
    return (
        f"For this blind defense, the call costs {price} into {pot} chips, requiring {price_share:.1%} "
        f"of the post-call pot. Simulated heads-up check-down comparisons: {comparison}. "
        "The tight, medium, and wide priors cover about 11%, 25%, and 43% of hands before blockers. "
        "Account for position, future bets, and placement before deciding."
    )
