"""Fold versus call at heads-up showdowns, using public commitments and sampled opponent ranges."""

from __future__ import annotations

import random
import re
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache
from typing import Any

from pokerkit import Deck, StandardHighHand

if __package__:
    from . import price_card
else:
    import price_card

DECK = {repr(card): card for card in Deck.STANDARD}
CALL = re.compile(r"Legal actions: fold, call (\d+)")
PROFILES = (0.65, 1.0, 1.6)
MIN_EDGE = 0.025


@dataclass(frozen=True)
class Call:
    hero: int
    villain: int
    hole: tuple[str, str]
    board: tuple[str, ...]
    street: str
    stack: tuple[int, ...]
    paid: tuple[int, ...]
    price: int
    hand: int
    hands: int
    button: int
    facing: bool
    late: bool
    shove: bool

    @property
    def live(self) -> int:
        return sum(s + paid > 0 for s, paid in zip(self.stack, self.paid, strict=True))


def late_position(seat: int, button: int, seats: Iterable[int]) -> bool:
    order = sorted(seats, key=lambda s: (s <= button, s))
    return seat in {order[0], button} or (len(order) >= 4 and seat == order[-2])


def public_call(
    messages: list[Any], starting: dict[int, int], aggressor: int, facing: bool
) -> Call | None:
    """Use only visible state. Starting stacks are announced at the beginning of the hand."""
    parsed = price_card.parse_state(messages)
    if parsed is None:
        return None
    state, rest = parsed
    call = CALL.search(rest)
    active = [s for s in state["seats"] if s["status"] not in ("out", "folded")]
    if not call or len(active) != 2:
        return None
    hero, villain = sorted(active, key=lambda s: not s["you"])
    price = int(call[1])
    if not hero["you"] or villain["you"] or villain["seat"] != aggressor:
        return None
    if price <= 0 or price != min(hero["stack"], villain["bet"] - hero["bet"]):
        return None
    if state["street"] != "river" and hero["stack"] > price and villain["stack"] > 0:
        return None
    seats = {s["seat"]: s for s in state["seats"]}
    if set(seats) != set(range(6)):
        return None
    stack = tuple(seats[i]["stack"] for i in range(6))
    paid = tuple(starting.get(i, 0) - stack[i] for i in range(6))
    if min(paid) < 0 or sum(paid) != state["pot"]:
        return None
    if max(paid) > max(paid[hero["seat"]] + price, paid[aggressor]):
        return None
    buttons = [s["seat"] for s in state["seats"] if s["button"]]
    if len(buttons) != 1 or len(state["hole"]) != 2:
        return None
    visible = state["hole"] + state["board"]
    if len(set(visible)) != len(visible) or any(c not in DECK for c in visible):
        return None
    return Call(
        hero["seat"],
        aggressor,
        tuple(state["hole"]),
        tuple(state["board"]),
        state["street"],
        stack,
        paid,
        price,
        state["hand"],
        state["hands"],
        buttons[0],
        facing,
        late_position(aggressor, buttons[0], starting),
        villain["stack"] == 0,
    )


def finish_stacks(spot: Call, result: int | None) -> tuple[int, ...]:
    """Settle a fold, win, loss or tie. Contributions include dead money from folded seats."""
    stacks = list(spot.stack)
    if result is None:
        stacks[spot.villain] += sum(spot.paid)
        return tuple(stacks)
    stacks[spot.hero] -= spot.price
    paid = list(spot.paid)
    paid[spot.hero] += spot.price
    previous, contested = 0, 0
    for level in sorted(set(paid) - {0}):
        contributors = [seat for seat, chips in enumerate(paid) if chips >= level]
        amount = (level - previous) * len(contributors)
        previous = level
        eligible = [seat for seat in (spot.hero, spot.villain) if paid[seat] >= level]
        if len(eligible) == 1:
            stacks[eligible[0]] += amount
        else:
            contested += amount
    if result:
        stacks[spot.hero if result > 0 else spot.villain] += contested
    else:
        half, odd = divmod(contested, 2)
        stacks[spot.hero] += half
        stacks[spot.villain] += half
        first = min((spot.hero, spot.villain), key=lambda s: (s <= spot.button, s))
        stacks[first] += odd
    return tuple(stacks)


def placement_value(spot: Call, stacks: tuple[int, ...], table: dict) -> float:
    hero = stacks[spot.hero]
    if hero == 0:
        return price_card.PLACE_POINTS[spot.live]
    live = sum(s > 0 for s in stacks)
    if spot.hand == spot.hands:
        return price_card.PLACE_POINTS[1 + sum(s > hero for s in stacks)]
    return price_card.stack_value(live, spot.hand, hero, table)


def values(spot: Call, table: dict) -> tuple[float, dict[int, float]]:
    folded = placement_value(spot, finish_stacks(spot, None), table)
    return folded, {
        i: placement_value(spot, finish_stacks(spot, i), table) - folded
        for i in (-1, 0, 1)
    }


@cache
def preflop_bucket(hole: tuple[str, str]) -> str:
    return price_card.preflop_class(list(hole))


def hand_bucket(hole: tuple[str, str], board: tuple[str, ...]) -> str:
    return (
        price_card.postflop_class(list(hole), list(board))
        if board
        else preflop_bucket(hole)
    )


def context(street: str, facing: bool, late: bool, bucket: str) -> tuple[str, str]:
    parent = f"{street}|{int(facing)}|{bucket}"
    return parent, f"{parent}|{int(late)}"


def bet_weight(
    spot: Call, bucket: str, book: dict, *, use_pluribus: bool = True
) -> float:
    """Shrink the field likelihood toward Pluribus for ordinary bets, with a weaker prior for shoves."""
    parent, key = context(spot.street, spot.facing, spot.late, bucket)
    strength = 12
    estimate = 0.12 if spot.shove else 0.25
    if use_pluribus:
        played, raised = book["pluribus"].get(parent, (0, 0))
        if played:
            observed = (raised + 2) / (played + 8)
            estimate = observed if not spot.shove else 0.5 * observed
    for k in (parent, key):
        played, bet, shove = book["arena"].get(k, (0, 0, 0))
        aggressive = shove if spot.shove else bet
        estimate = (aggressive + strength * estimate) / (played + strength)
    return estimate


def outcome(spot: Call, villain: tuple[str, str], board: tuple[str, ...]) -> int:
    public = tuple(DECK[c] for c in board)
    hero_rank = StandardHighHand.from_game(tuple(DECK[c] for c in spot.hole), public)
    villain_rank = StandardHighHand.from_game(tuple(DECK[c] for c in villain), public)
    return int(hero_rank > villain_rank) - int(hero_rank < villain_rank)


def worlds(spot: Call, samples: int, seed: str) -> list[tuple[str, int]]:
    rng = random.Random(seed)
    deck = [c for c in DECK if c not in spot.hole and c not in spot.board]
    return [
        (hand_bucket((a, b), spot.board), outcome(spot, (a, b), (*spot.board, *extra)))
        for a, b, *extra in (
            rng.sample(deck, 7 - len(spot.board)) for _ in range(samples)
        )
    ]


def estimate(
    spot: Call,
    simulations: list[tuple[str, int]],
    book: dict,
    gains: dict[int, float],
    *,
    use_pluribus: bool = True,
) -> tuple[float, ...]:
    by_bucket = {}
    for bucket, _ in simulations:
        if bucket not in by_bucket:
            weight = bet_weight(spot, bucket, book, use_pluribus=use_pluribus)
            by_bucket[bucket] = tuple(weight**temp for temp in PROFILES)
    totals, amounts = [0.0] * len(PROFILES), [0.0] * len(PROFILES)
    for bucket, result in simulations:
        for i, weight in enumerate(by_bucket[bucket]):
            totals[i] += weight
            amounts[i] += weight * gains[result]
    return tuple(a / n for a, n in zip(amounts, totals, strict=True))


def robust_action(edges: tuple[float, ...], margin: float = MIN_EDGE) -> str | None:
    if min(edges) > margin:
        return "call"
    if max(edges) < -margin:
        return "fold"
    return None
