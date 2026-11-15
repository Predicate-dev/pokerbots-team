"""Placement value: what a stack is worth when the game is ranked by finishing place, not by chips.

The arena rates an entry on where it finishes each 30-hand game: seats still in rank by chips, above every busted
seat. So first by 1 chip equals first by 400, a lead late in the game is worth protecting, and going bust is the
worst result. `place_value` turns a set of stacks and the number of hands left into the share of the other seats
the hero is expected to finish above (1.0 = certain first, 0.0 = certain last), so a hand can be rewarded by how
much it moved that number (`grading.py`, `grader_params.reward = "placement"`).

The model is deliberately simple and closed-form. Against each other seat, the hero finishes above with
probability Phi(lead / (noise * sqrt(hands_left))): the gap between two stacks drifts like a random walk over the
remaining hands, with `noise_chips` its standard deviation per hand. With no hands left it is the exact ranking.
A seat with no chips is out: it cannot come back, and it ranks below every seat still in.

What this ignores: skill differences, position, and that a short stack's swings are smaller than a big stack's. It
is a reward shape, not a solver.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

DEFAULT_NOISE_BB = 12.0  # per-hand standard deviation of the gap between two seats' stacks, in big blinds


def _beats(lead: float, spread: float) -> float:
    """P(the hero finishes above a seat it leads by `lead` chips), given the spread of that gap by the end."""
    if spread <= 0:
        return 1.0 if lead > 0 else 0.5 if lead == 0 else 0.0
    return 0.5 * (1.0 + math.erf(lead / (spread * math.sqrt(2.0))))


def place_value(
    stacks: Mapping[str, int],
    hero: str,
    hands_left: int,
    noise_chips: float,
    before: Mapping[str, int] | None = None,
) -> float:
    """The share of the other seats `hero` is expected to finish above, in [0, 1].

    `stacks` has every seat at the table (0 = out). `hands_left` is how many hands are still to be played from
    these stacks. `before` is each seat's stack before the hand that produced `stacks`; it only orders seats that
    are both out, the way the event does: a seat already out before the hand ranks below one that busted in it, and
    seats that busted in the same hand rank by the stack they started it with."""
    others = [name for name in stacks if name != hero]
    if hero not in stacks or not others:
        raise ValueError(f"{hero!r} needs a stack and at least one other seat")
    mine = stacks[hero]
    spread = noise_chips * math.sqrt(max(0, hands_left))
    total = 0.0
    for name in others:
        theirs = stacks[name]
        if mine > 0 and theirs > 0:
            total += _beats(mine - theirs, spread)
        elif mine > 0 or theirs > 0:  # exactly one of the two is out
            total += 1.0 if mine > 0 else 0.0
        elif before is not None:  # both out: later bust, then bigger stack before it, ranks higher
            total += _beats(before.get(hero, 0) - before.get(name, 0), 0.0)
        else:
            total += 0.5
    return total / len(others)
