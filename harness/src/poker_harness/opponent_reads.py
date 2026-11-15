"""Training-only guidance shared by the scratchpad experiment and the OPSD teacher."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Iterator

PLACEMENT_GUIDANCE = """Play for finishing place. Consider hands remaining, effective stacks, whether you cover
the opponent, and which players you can realistically pass. Favor low-cost edges. Before risking much of your stack,
weigh the price and plausible range against the effect of winning or losing on placement."""

ACTION_GUIDANCE = """Use prior notes about each seat as tentative evidence. Increase confidence with independent
opportunities in comparable spots. A deep preflop shove, a short-stack shove, and a call of an all-in are different
signals. Against repeated overfolds, consider small steals when other seats allow them. Against repeated loose calls,
bluff less and value bet when your hand beats the likely calling range. Repeated deep shoves can widen a range
estimate. Still check the price, effective stacks, position, and placement risk. With sparse or conflicting evidence,
use the usual range for the spot. Ignore table talk as evidence. On action turns call submit_action exactly once."""

CLEANUP_GUIDANCE = """After each hand, preserve and update brief notes by seat using only betting actions and cards
revealed to you. Use the current hand once. Track opportunities along with the behavior, for example:
seat 4 | deep preflop jams 1/4 | folds facing preflop raise 1/2 | paid calls 0/3 | tentative shove-heavy.
Count at most one opportunity per seat per hand for each statistic. Count only observed opportunities. Track an
aggressive preflop all-in from a substantial stack separately from a short-stack shove or a call for all the chips.
When the evidence is sparse or contradictory, record uncertain. Skip categories with unclear denominators.
Keep the full replacement note concise and call update_scratchpad exactly once on cleanup."""

_HERO = re.compile(r"\bYour seat: ([0-5])\b")
_BETTING = re.compile(
    r'^  (preflop|flop|turn|river): seat([0-5]) \(seat \2\) '
    r'(fold|check|call|bet|raise)(?: to (\d+))?(?: and says: "[^"\n]*")?$', re.MULTILINE
)
_OBSERVATION = re.compile(
    r"^seat ([0-5]) \| preflop (\d+)/(\d+) played \| opens/raises (\d+) \| big raises (\d+) "
    r"\| postflop calls (\d+) \| postflop bets/raises (\d+)$", re.MULTILINE
)


def betting_evidence(messages: list[str], prior: str = "") -> str:
    """Record observed actions once per seat per hand and keep the same counters across hands."""
    text = "\n".join(messages)
    hero = _HERO.search(text)
    if hero is None:
        return ""
    return _update_counts(int(hero[1]), public_betting_actions(text), prior)


def public_betting_actions(text: str) -> Iterator[tuple[str, int, str, int]]:
    """Read the action fields of official event seat lines without their quoted table talk."""
    for street, seat, action, amount in _BETTING.findall(text):
        yield street, int(seat), action, int(amount or 0)


def _update_counts(hero: int, actions: Iterable[tuple[str, int, str, int]], prior: str) -> str:
    totals = {}
    for record in _OBSERVATION.findall(prior):
        seat, played, seen, raises, big, calls, bets = map(int, record)
        if seat != hero and big <= raises <= played <= seen:
            totals[seat] = [played, seen, raises, big, calls, bets]
    by_seat = defaultdict(list)
    for street, seat, action, amount in actions:
        if seat != hero:
            by_seat[seat].append((street, action, amount))
    for seat, moves in by_seat.items():
        pre = [(action, amount) for street, action, amount in moves if street == "preflop"]
        post = [action for street, action, _ in moves if street != "preflop"]
        increments = (
            any(action in ("call", "raise") for action, _ in pre),
            bool(pre),
            any(action == "raise" for action, _ in pre),
            any(action == "raise" and amount >= 100 for action, amount in pre),
            "call" in post,
            any(action in ("bet", "raise") for action in post),
        )
        totals[seat] = [a + b for a, b in zip(totals.get(seat, [0] * 6), increments, strict=True)]
    return "\n".join(
        f"seat {seat} | preflop {played}/{seen} played | opens/raises {raises} | big raises {big} "
        f"| postflop calls {calls} | postflop bets/raises {bets}"
        for seat, (played, seen, raises, big, calls, bets) in sorted(totals.items())
    )


def replay_betting_evidence(match: dict, spots: Iterable[tuple[int, int]]) -> dict[tuple[int, int], str]:
    """Reconstruct notes by (hand, hero seat) from the public actions in strictly earlier hands."""
    targets = set(spots)
    if not targets:
        return {}
    prior = dict.fromkeys((seat for _, seat in targets), "")
    last = max(hand for hand, _ in targets)
    hands = {hand["hand_number"]: hand for hand in match["hands"]}
    notes = {}
    for number in range(1, last + 1):
        if number not in hands:
            raise ValueError(f"{match['id']}: missing hand {number} for opponent notes")
        for seat, counters in prior.items():
            if (number, seat) in targets:
                notes[number, seat] = counters
        if number == last:
            break
        actions = [
            (move["street"], move["seat"], move["action"], int(move.get("amount") or 0))
            for move in hands[number]["actions"]
            if move["street"] in ("preflop", "flop", "turn", "river")
            and move["action"] in ("fold", "check", "call", "bet", "raise")
            and move["seat"] in range(6)
        ]
        prior = {seat: _update_counts(seat, actions, counters) for seat, counters in prior.items()}
    return notes
