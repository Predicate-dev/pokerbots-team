"""Which playbook rule (`opsd/notes/playbook.txt`) a decision falls under, and whether the reply follows it.

    from playbook_rules import verdicts
    verdicts(prompt_items, {"action": "raise", "amount": 5})   # -> [("A", True)]

Used to pick training targets (`build_checklist_sft.py --playbook`) and to count how often a model follows each rule.
A decision can fall under no rule, in which case the list is empty. Everything is read from the decision prompt
alone: the public state, the actions listed, and this seat's own earlier calls in the hand.

- A  everyone has folded to this seat before the flop: raise or fold by position, never just call.
- B  this seat raised before the flop and it is checked to it on the flop: bet.
- C  more than 20 chips into a pot only with a hand on the list (violations only).
- D  facing one ordinary raise with QQ+, AK, 99-JJ or AQ: re-raise. Own raise re-raised: continue only with QQ+ or AK.
"""

from __future__ import annotations

import json
import re
from typing import Any

import open_card
from price_card import parse_state, postflop_class, preflop_class

RANKS = "23456789TJQKA"
BIG_POT = 20
ORDINARY_RAISE = 12  # a raise to more than this is a shove or an over-bet, which the checklist prices instead
_RAISE = re.compile(r"preflop: [^\n]*? raise to (\d+)")
_STREET = re.compile(r"Street: (\w+)")
_SEAT = re.compile(r"\(you\) stack (\d+), bet (\d+)")
_CALL = re.compile(r"call (\d+)")


def opens(position: str, hole: list[str]) -> bool:
    """Rule A: does this hand raise from this position when everyone has folded?"""
    if position in ("SB", "BTN"):
        return True
    high, low = sorted((RANKS.index(c[0]) for c in hole), reverse=True)
    pair, suited = high == low, hole[0][1] == hole[1][1]
    if position == "CO":
        return pair or high == 12 or low >= 8 or (suited and high - low == 1)
    return (pair and high >= 7) or (high == 12 and low >= 9) or (high == 11 and low == 10)


def _content(item: Any) -> str:
    return item.get("content") if isinstance(item.get("content"), str) else ""


def own_actions(prompt: list[dict]) -> list[tuple[str, str]]:
    """(street, action) for each action this seat already took in the hand."""
    street, out = "preflop", []
    for item in prompt:
        if item.get("role") == "user":
            for found in _STREET.finditer(_content(item)):
                street = found[1]
        elif item.get("type") == "function_call" and item.get("name") == "submit_action":
            try:
                out.append((street, json.loads(item["arguments"]).get("action")))
            except (TypeError, ValueError):
                continue
    return out


def verdicts(prompt: list[dict], arguments: dict) -> list[tuple[str, bool]]:
    parsed = parse_state(prompt)
    if parsed is None:
        return []
    spot, rest = parsed
    action = arguments.get("action")
    hole, street = spot["hole"], spot["street"]
    earlier = own_actions(prompt)
    users = [_content(m) for m in prompt if m.get("role") == "user"]
    out: list[tuple[str, bool]] = []
    hand = preflop_class(hole)
    queens_up = hand in ("QQ+", "AK")
    if street == "preflop":
        first_in = open_card.parse(prompt)
        raises = [int(n) for n in _RAISE.findall(users[-1])]
        if first_in is not None and not earlier and first_in["stack"] >= BIG_POT:
            if action == "call":
                out.append(("A", False))
            else:
                out.append(("A", (action == "raise") == opens(first_in["position"], hole)))
        elif not earlier and len(raises) == 1 and raises[0] <= ORDINARY_RAISE:
            ace_queen = sorted(c[0] for c in hole) == ["A", "Q"]
            if queens_up or hand == "99-JJ" or ace_queen:
                out.append(("D", action == "raise"))
        elif any(a == "raise" for s, a in earlier if s == "preflop") and raises and "Since your last action" in users[-1]:
            out.append(("D", (action != "fold") == queens_up))
    elif street == "flop" and "Legal actions: check" in rest:
        raised = any(a == "raise" for s, a in earlier if s == "preflop")
        if raised and not any(s == "flop" for s, _ in earlier):
            out.append(("B", action in ("bet", "raise")))
    # C: chips in the pot from this seat after this action
    first = next((_SEAT.search(u) for u in users if _SEAT.search(u)), None)
    now = _SEAT.search(users[-1][: -len(rest)] if rest else users[-1])
    if first and now and action in ("call", "bet", "raise"):
        start = int(first[1]) + int(first[2])
        if action == "call":
            found = _CALL.search(rest)
            added = min(int(found[1]), int(now[1])) if found else 0
        else:
            added = int(arguments.get("amount") or 0) - int(now[2])
        total = start - int(now[1]) + added
        if total > BIG_POT:
            strong = queens_up if street == "preflop" else postflop_class(hole, spot["board"]) in ("monster", "trips", "two_pair", "overpair")
            if not strong:
                out.append(("C", False))
    return out
