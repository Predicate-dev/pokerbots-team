"""The seat ledger: a fixed layout for the scratchpad, with one line of counters per seat.

A game is 30 hands, and counts are the only read that firms up that fast. Free-form notes drift every time the
model rewrites them; a fixed layout turns the after-hand update into adding to numbers. The ledger explains itself
(`GOAL` and `KEY` lines), so a model that sees one in its scratchpad can keep it going, and since the scratchpad is
the only text a team controls at the event, those lines are also where standing advice lives.

    LEDGER v1 | after hand 26 of 30 | me seat3
    GOAL ...
    KEY ...
    MODES ...
    MODE small through hand 30
    seat0 stack 155 | n26 vp9 pr6 3b1 ab7 pc4 fd5 sd2 | open 5,5,6 || tight, folds to raises
    seat2 out
    seat3 stack 218 (me) | n26 vp8 pr7 3b1 ab6 pc2 fd3 sd1 | open 5,5,5 || shown AKo once
    ...any other lines are the model's own notes...

Everything before `||` on a seat line is public arithmetic, so `update` can compute the right answer from a
finished hand and `accuracy` can grade a model's ledger against it (`grading.py`, `grader_params.ledger_weight`).
Everything after `||`, and every other line, belongs to the model and is never graded. The MODE line names the
persona in play (`persona.py`); it also has one right answer, graded on its own (`grader_params.persona_weight`).

Counters, each from what every seat can see:
    n   hands the seat was dealt into
    vp  hands it paid to play before the flop (a call or a raise; posting a blind or checking it does not count)
    pr  hands it raised before the flop
    3b  hands it raised before the flop after someone else had already raised
    ab  bets and raises it made after the flop
    pc  calls it made after the flop
    fd  hands it folded after the flop
    sd  showdowns it reached
    open  its last three raise-to sizes when it was the first to raise before the flop
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .persona import MODES, RULE, next_mode

VERSION = "LEDGER v1"
COUNTERS = ("n", "vp", "pr", "3b", "ab", "pc", "fd", "sd")
MAX_OPENS = 3
GOAL = (
    "GOAL best finishing place, not most chips. Ahead late: protect the lead. Behind late: gamble. "
    "Busting is last place."
)
KEY = (
    "KEY per seat, add after every hand: n dealt | vp paid to play preflop | pr raised preflop | 3b reraised "
    "preflop | ab postflop bets+raises | pc postflop calls | fd postflop folds | sd showdowns | open last 3 "
    "raise-to sizes when first to raise | after || my own read of the seat"
)

_HEADER = re.compile(rf"{re.escape(VERSION)}\s*\|\s*after hand (\d+) of (\d+)\s*\|\s*me (\w+)")
_MODE = re.compile(rf"^\s*MODE ({'|'.join(MODES)}) through hand (\d+)\s*$")
_SEAT = re.compile(
    r"^\s*(?P<name>seat\d+)\s+(?:(?P<out>out)\b.*|stack (?P<stack>\d+)(?: \(me\))?\s*\|\s*"
    + r"\s+".join(rf"{re.escape(c)}(?P<c{i}>\d+)" for i, c in enumerate(COUNTERS))
    + r"\s*\|\s*open\s*(?P<opens>[\d,\s]*|-)\s*(?:\|\|(?P<note>.*))?)$"
)


@dataclass
class SeatLine:
    stack: int = 0  # 0 = out
    counts: dict[str, int] = field(default_factory=lambda: dict.fromkeys(COUNTERS, 0))
    opens: list[int] = field(default_factory=list)
    note: str = ""


@dataclass
class Ledger:
    hand: int  # the last hand counted
    total: int
    me: str
    seats: dict[str, SeatLine]
    notes: list[str] = field(default_factory=list)  # the model's own lines, kept as they are
    mode: str | None = None  # the persona in play (`persona.py`), and the last hand it is held for
    mode_through: int = 0


def render(ledger: Ledger) -> str:
    lines = [f"{VERSION} | after hand {ledger.hand} of {ledger.total} | me {ledger.me}", GOAL, KEY, RULE]
    if ledger.mode:
        lines.append(f"MODE {ledger.mode} through hand {ledger.mode_through}")
    for name, seat in ledger.seats.items():
        if seat.stack <= 0:
            lines.append(f"{name} out")
            continue
        counts = " ".join(f"{c}{seat.counts[c]}" for c in COUNTERS)
        opens = ",".join(str(size) for size in seat.opens) or "-"
        me = " (me)" if name == ledger.me else ""
        lines.append(
            f"{name} stack {seat.stack}{me} | {counts} | open {opens} ||{' ' + seat.note if seat.note else ''}"
        )
    return "\n".join(lines + ledger.notes)


def parse(text: str) -> Ledger | None:
    """The ledger in a scratchpad, or None when it has no ledger header. Seat lines that don't fit the layout are
    kept as notes, so a slip on one line costs that line and not the rest."""
    header = _HEADER.search(text or "")
    if not header:
        return None
    ledger = Ledger(int(header[1]), int(header[2]), header[3], {})
    for line in text[header.end() :].splitlines():
        match = _SEAT.match(line)
        mode = _MODE.match(line)
        if mode and ledger.mode is None:
            ledger.mode, ledger.mode_through = mode[1], int(mode[2])
        elif match and match["name"] not in ledger.seats:
            if match["out"]:
                ledger.seats[match["name"]] = SeatLine()
                continue
            opens = [int(size) for size in re.findall(r"\d+", match["opens"] or "")]
            counts = {c: int(match[f"c{i}"]) for i, c in enumerate(COUNTERS)}
            note = (match["note"] or "").strip()
            ledger.seats[match["name"]] = SeatLine(int(match["stack"]), counts, opens, note)
        elif line.strip() and not line.startswith(("GOAL", "KEY", "MODES")):
            ledger.notes.append(line)
    return ledger


def from_params(params: dict[str, Any]) -> Ledger:
    """A ledger from a task's `env_params.starting_ledger`: {"hand", "total", "me", "seats": {name: {"stack", the
    counters, "open": [...], "note"}}, "notes": [...], "mode": {"name", "through"}}. Missing counters are 0; a stack
    of 0 is a seat that is out; without "mode" there is no MODE line."""
    seats = {}
    for name, row in params["seats"].items():
        counts = {c: int(row.get(c, 0)) for c in COUNTERS}
        opens = [int(size) for size in row.get("open", [])][-MAX_OPENS:]
        seats[name] = SeatLine(int(row.get("stack", 0)), counts, opens, str(row.get("note", "")))
    notes = [str(line) for line in params.get("notes", [])]
    mode = params.get("mode") or {}
    if mode and mode.get("name") not in MODES:
        raise ValueError(f"starting_ledger.mode.name must be one of {MODES}")
    return Ledger(
        int(params["hand"]),
        int(params["total"]),
        str(params["me"]),
        seats,
        notes,
        mode.get("name"),
        int(mode.get("through", 0)),
    )


def to_params(ledger: Ledger) -> dict[str, Any]:
    """The reverse of `from_params`: the ledger as a task's `env_params.starting_ledger`."""
    seats: dict[str, dict[str, Any]] = {}
    for name, seat in ledger.seats.items():
        if seat.stack <= 0:
            seats[name] = {"stack": 0}
            continue
        seats[name] = {"stack": seat.stack, **seat.counts, "open": list(seat.opens)}
        if seat.note:
            seats[name]["note"] = seat.note
    out: dict[str, Any] = {"hand": ledger.hand, "total": ledger.total, "me": ledger.me, "seats": seats}
    if ledger.mode:
        out["mode"] = {"name": ledger.mode, "through": ledger.mode_through}
    return out | ({"notes": list(ledger.notes)} if ledger.notes else {})


def update(prior: Ledger | None, hand: dict[str, Any], me: str, total: int) -> Ledger:
    """The ledger after `hand` (a game log's hand record): `prior` with this hand's public actions added. Without a
    usable `prior` every seat starts from zero. Notes and reads carry over untouched. The MODE line is kept while
    its block lasts and otherwise drawn from `me`'s first hole card this hand (`persona.next_mode`)."""
    names = list(hand["ending_stacks"])
    carried = prior is not None and prior.me == me
    seats = {}
    for name in names:
        old = prior.seats.get(name) if carried else None
        seats[name] = SeatLine(0, dict(old.counts), list(old.opens), old.note) if old else SeatLine()
    dealt = set(hand["starting_state"])
    for name in dealt:
        seats[name].counts["n"] += 1
    flags: dict[str, set[str]] = {c: set() for c in ("vp", "pr", "3b", "fd")}
    folded: set[str] = set()
    raises = 0
    for action in hand["actions"]:
        name, kind = action["player"], action["action"]["type"]
        if kind == "fold":
            folded.add(name)
        if action["street"] == "preflop":
            if kind in ("bet", "raise"):
                flags["vp"].add(name)
                flags["pr"].add(name)
                if raises:
                    flags["3b"].add(name)
                else:
                    seats[name].opens = [*seats[name].opens, int(action["action"]["amount"])][-MAX_OPENS:]
                raises += 1
            elif kind == "call":
                flags["vp"].add(name)
        elif kind in ("bet", "raise"):
            seats[name].counts["ab"] += 1
        elif kind == "call":
            seats[name].counts["pc"] += 1
        elif kind == "fold":
            flags["fd"].add(name)
    for counter, who in flags.items():
        for name in who:
            seats[name].counts[counter] += 1
    if hand.get("showdown"):
        for name in dealt - folded:
            seats[name].counts["sd"] += 1
    for name in names:
        seats[name].stack = hand["ending_stacks"][name]
    notes = list(prior.notes) if carried else []
    held = (prior.mode, prior.mode_through) if carried else (None, 0)
    hole = hand["starting_state"].get(me, {}).get("hole_cards") or []
    mode, through = next_mode(*held, hand["hand_number"], hole)
    return Ledger(hand["hand_number"], total, me, seats, notes, mode, through)


def accuracy(text: str, reference: Ledger) -> float:
    """The share of the reference ledger's graded facts that `text` gets right, in [0, 1]: the hand number, and for
    every seat its stack (or that it is out), each counter and its open sizes. No ledger header scores 0."""
    ledger = parse(text)
    if ledger is None:
        return 0.0
    right, total = int(ledger.hand == reference.hand and ledger.me == reference.me), 1
    for name, want in reference.seats.items():
        got = ledger.seats.get(name)
        if want.stack <= 0:  # an out seat is one fact
            right, total = right + int(got is not None and got.stack <= 0), total + 1
            continue
        total += 2 + len(COUNTERS)
        if got is None or got.stack <= 0:
            continue
        right += int(got.stack == want.stack) + int(got.opens == want.opens)
        right += sum(got.counts[c] == want.counts[c] for c in COUNTERS)
    return right / total
