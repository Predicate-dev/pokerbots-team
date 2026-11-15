"""The equal-rated OpenSkill placement proxy and its expected value from current stacks."""

from __future__ import annotations

import math
from collections.abc import Mapping
from functools import cache
from statistics import NormalDist

DEFAULT_NOISE_BB = 12.0  # per-hand standard deviation of the gap between two seats' stacks, in big blinds
_NORMAL = NormalDist()
_HERO_SHIFTS = tuple(_NORMAL.inv_cdf((i + 0.5) / 32) for i in range(32))


def rating_points(place: int, seats: int = 6) -> float:
    """Relative Plackett-Luce update at a table of equally rated players."""
    return 1 - sum(1 / (seats + 1 - rank) for rank in range(1, place + 1))


@cache
def rating_weights(seats: int) -> tuple[float, ...]:
    """The same placement curve rescaled from last at 0 to first at 1."""
    if seats == 1:
        return (1.0,)
    points = [rating_points(place, seats) for place in range(1, seats + 1)]
    return tuple((point - points[-1]) / (points[0] - points[-1]) for point in points)


def place_value(
    stacks: Mapping[str, int],
    hero: str,
    hands_left: int,
    noise_chips: float,
    before: Mapping[str, int] | None = None,
) -> float:
    """The expected rating weight of a seat's finishing rank, in [0, 1].

    Model future stack gaps with independent Gaussian shocks to each live seat. Conditional on
    the hero's stack, convolve the chances of an opponent passing it. Averaging over the hero's
    shock preserves the correlation between those events. Busted seats stay below live seats.
    For seats that bust together, compare the stacks they started the hand with.
    """
    if hero not in stacks or len(stacks) < 2:
        raise ValueError(f"{hero!r} needs a stack and at least one other seat")
    mine = stacks[hero]
    weights = rating_weights(len(stacks))
    if mine <= 0:
        prior = before or {}
        my_prior = prior.get(hero, 0)
        ahead = sum(stack > 0 or prior.get(name, 0) > my_prior for name, stack in stacks.items() if name != hero)
        return weights[ahead]
    others = [stack for name, stack in stacks.items() if name != hero and stack > 0]
    spread = noise_chips * math.sqrt(max(0, hands_left) / 2)
    if not others or spread <= 0:
        return weights[sum(stack > mine for stack in others)]
    if all(stack == mine for stack in others):
        return sum(weights[: len(others) + 1]) / (len(others) + 1)

    gaps = [(stack - mine) / spread for stack in others]
    expected = 0.0
    for shift in _HERO_SHIFTS:
        ranks = [1.0]
        for gap in gaps:
            passes = _NORMAL.cdf(gap - shift)
            ranks.append(0.0)
            for index in range(len(ranks) - 1, 0, -1):
                ranks[index] = ranks[index] * (1 - passes) + ranks[index - 1] * passes
            ranks[0] *= 1 - passes
        expected += sum(probability * weight for probability, weight in zip(ranks, weights))
    return expected / len(_HERO_SHIFTS)
