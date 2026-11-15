"""A rating-aware raise-or-fold card for an unopened preflop pot in the 1/2 arena."""

from __future__ import annotations

import json
import re
from functools import cache
from pathlib import Path
from typing import Any

if __package__:
    from .price_card import PREFLOP_NAMES, RANKS, parse_state, preflop_class
else:
    from price_card import PREFLOP_NAMES, RANKS, parse_state, preflop_class

TABLES = Path(__file__).with_name("open_tables.json")
MARK = "Your read of this opening decision."
MIN_OPEN_TO, OPEN_TO, MAX_OPEN_TO = 4, 5, 6  # 2.5bb target, trained on 2–3bb opens
SHORT_STACK = 20  # 10bb effective
MIN_SAMPLES = 8
SHRINK = 12
CLOSE = 0.003  # three rating points per 1,000 decisions

_CHOICE = re.compile(
    r"Actions this hand:\n(?P<actions>.*?)\n\nLegal actions: fold, call (?P<call>\d+), "
    r"raise to (?P<low>\d+)-(?P<high>\d+)\nCall submit_action with your decision\.\Z",
    re.DOTALL,
)
_FOLD = re.compile(r'  preflop: [^\n]+? \(seat \d+\) fold(?: and says: "[^"\n]*")?')
_CARD = re.compile(r"[2-9TJQKA][cdhs]")

NAMES = PREFLOP_NAMES | {
    "Axs": "a suited ace with a weaker kicker",
    "Axo": "an offsuit ace with a weaker kicker",
    "broadway_s": "two suited broadway cards without an ace",
    "broadway_o": "two offsuit broadway cards without an ace",
    "suited_connector": "suited consecutive ranks",
}


def hand_class(hole: list[str]) -> str:
    group = preflop_class(hole)
    suited = hole[0][1] == hole[1][1]
    if group == "Ax":
        return "Axs" if suited else "Axo"
    if group == "broadway":
        return "broadway_s" if suited else "broadway_o"
    if (
        group == "suited"
        and abs(RANKS.index(hole[0][0]) - RANKS.index(hole[1][0])) == 1
    ):
        return "suited_connector"
    return group


def parse(messages: list[Any]) -> dict[str, Any] | None:
    parsed = parse_state(messages)
    if parsed is None:
        return None
    spot, rest = parsed
    if spot["street"] != "preflop" or spot["board"]:
        return None
    hole = spot["hole"]
    choice = _CHOICE.fullmatch(rest)
    if (
        choice is None
        or hole[0] == hole[1]
        or any(not _CARD.fullmatch(c) for c in hole)
    ):
        return None
    prior = choice["actions"]
    if prior != "  (no actions yet)" and (
        not prior or not all(_FOLD.fullmatch(line) for line in prior.splitlines())
    ):
        return None
    seats = [s for s in spot.pop("seats") if s["status"] != "out"]
    buttons = [s["seat"] for s in seats if s["button"]]
    heroes = [s for s in seats if s["you"] and not s["status"]]
    if len(buttons) != 1 or len(heroes) != 1 or not 2 <= len(seats) <= 6:
        return None
    me, button = heroes[0], buttons[0]
    order = sorted((s["seat"] for s in seats), key=lambda seat: (seat <= button, seat))
    index = order.index(me["seat"])
    if len(order) == 2:
        if me["seat"] != button:
            return None  # the button posts the small blind heads-up
        position = "SB"
    elif index == 0:
        position = "SB"
    elif index == 1:
        return None  # the big blind has a free check in an unopened pot
    else:
        position = ("BTN", "CO", "HJ", "UTG")[len(order) - 1 - index]
    low, high = int(choice["low"]), int(choice["high"])
    to_call = int(choice["call"])
    if not 0 < to_call < me["stack"] or not 0 < low <= high:
        return None
    opponents = [
        s["stack"] + s["bet"] for s in seats if not s["you"] and s["status"] != "folded"
    ]
    if not opponents:
        return None
    depth = min(me["stack"] + me["bet"], max(opponents))
    return spot | {
        "position": position,
        "hand_class": hand_class(hole),
        "depth": "short" if depth <= SHORT_STACK else "regular",
        "stack": me["stack"],
        "posted": me["bet"],
        "to_call": to_call,
        "min_raise_to": low,
        "max_raise_to": high,
        "raise_to": max(low, min(high, OPEN_TO)),
    }


def cell_key(spot: dict[str, Any]) -> str:
    return f"{spot['depth']}|{spot['position']}|{spot['live']}|{spot['hand_class']}"


def cell_keys(spot: dict[str, Any], action: str) -> list[str]:
    """Pool sparse raises by hand class and sparse folds among seats posting the same blind."""
    depth, pos, live, kind = (
        spot["depth"],
        spot["position"],
        spot["live"],
        spot["hand_class"],
    )
    if action == "raise":
        parents = [f"{depth}|*|*|{kind}", f"{depth}|{pos}|*|{kind}"]
    else:
        parents = [f"{depth}|{pos}|*|*", f"{depth}|{pos}|{live}|*"]
    return [*parents, cell_key(spot)]


def standard_raise(spot: dict[str, Any], amount: int) -> bool:
    return (
        max(spot["min_raise_to"], MIN_OPEN_TO)
        <= amount
        <= min(spot["max_raise_to"], MAX_OPEN_TO)
        and amount < spot["stack"] + spot["posted"]
    )


@cache
def tables() -> dict[str, Any]:
    return json.loads(TABLES.read_text())


def estimate(
    spot: dict[str, Any], action: str, table: dict[str, Any]
) -> tuple[float | None, int]:
    value, samples = None, 0
    for key in cell_keys(spot, action):
        cell = table[action].get(key)
        if cell is None:
            continue
        mean, count = cell
        value = (
            mean
            if value is None
            else (mean * count + value * SHRINK) / (count + SHRINK)
        )
        samples = max(samples, count)
    return value, samples


def plan(spot: dict[str, Any], table: dict[str, Any]) -> dict[str, Any]:
    raised, raises = estimate(spot, "raise", table)
    folded, folds = estimate(spot, "fold", table)
    edge = None if raised is None or folded is None else raised - folded
    supported = min(raises, folds) >= MIN_SAMPLES and standard_raise(
        spot, spot["raise_to"]
    )
    return spot | {
        "raise_value": raised,
        "fold_value": folded,
        "edge": edge,
        "raise_samples": raises,
        "fold_samples": folds,
        "plan": "raise" if supported and edge > 0 else "fold",
        "clear": supported and abs(edge) >= CLOSE,
    }


def card(
    messages: list[Any], table: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    spot = parse(messages)
    return (
        None
        if spot is None
        else plan(spot, (tables() if table is None else table)["open"])
    )


def answer(card: dict[str, Any], arguments: dict[str, Any]) -> str:
    action, amount = arguments.get("action"), arguments.get("amount")
    if action == "fold":
        return "fold"
    if action == "raise" and type(amount) is int and standard_raise(card, amount):
        return "raise"
    return "invalid"


def hint(c: dict[str, Any]) -> str:
    """Teacher-only text for a decision with enough evidence to teach."""
    if not c["clear"]:
        return ""
    action = f"raise to {c['raise_to']} chips" if c["plan"] == "raise" else "fold"
    return (
        f"{MARK} Use these estimates as your own reasoning.\n"
        f"- With {c['live']} players left you are in {c['position']} holding {NAMES[c['hand_class']]}.\n"
        f"- In comparable arena spots, a standard raise averaged {1000 * c['raise_value']:+.1f} rating points "
        f"per 1,000 decisions. Folding averaged {1000 * c['fold_value']:+.1f}.\n"
        f"- Plan: {action}. Choose the action once with submit_action."
    )
