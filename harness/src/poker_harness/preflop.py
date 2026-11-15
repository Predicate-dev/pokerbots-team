"""Observable preflop opportunities shared by live-replay and local scoring."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class PreflopAction:
    seat: int
    action: str
    to_call: int
    raise_to: int | None = None


def clockwise(seats: Iterable[int], button: int) -> list[int]:
    return sorted(seats, key=lambda seat: (seat <= button, seat))


def opportunities(
    actions: Iterable[PreflopAction],
    seats: Iterable[int],
    button: int,
    big_blind: int,
) -> Iterator[tuple[str, ...]]:
    """Tags at each decision, using only actions already taken. Free checks are excluded."""
    order = clockwise(seats, button)
    if len(order) < 2:
        return
    small, big = order[:2] if len(order) > 2 else (order[1], order[0])
    late = {small, button} | ({order[-2]} if len(order) >= 4 else set())
    live = set(order)
    raises: list[PreflopAction] = []
    limped = False
    for action in actions:
        tags = []
        if action.to_call > 0:
            if not raises:
                tags.append("unraised")
                if not limped and action.seat in late:
                    tags.append("late_first_in")
                    if action.seat == button:
                        tags.append("button_first_in")
            elif len(raises) > 1:
                tags.append("reraised")
            else:
                opener = raises[0]
                if opener.raise_to is not None and opener.raise_to <= 3 * big_blind:
                    tags.append("single_open_le_3bb")
                    if action.seat == big and opener.seat in late and live == {big, opener.seat}:
                        tags.append("bb_vs_late_small_open")
                else:
                    tags.append("single_open_gt_3bb")
        yield tuple(tags)
        if action.action == "fold":
            live.discard(action.seat)
        elif action.action in ("bet", "raise"):
            raises.append(action)
        elif action.action == "call" and action.to_call > 0 and not raises:
            limped = True


def record_folds(totals: dict[str, dict[str, int]], tags: Iterable[str], action: str) -> None:
    for tag in tags:
        bucket = totals.setdefault(tag, {"folds": 0, "opportunities": 0})
        bucket["opportunities"] += 1
        bucket["folds"] += action == "fold"
