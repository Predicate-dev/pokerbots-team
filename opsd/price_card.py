"""The price card: for a decision facing a bet, the three numbers that settle it, from what the seat can see.

    need  how often the call has to win to be right for the seat's finishing place (not for its chips)
    win   how often a hand like this has won in the arena against the hands that made this kind of bet
    plan  continue if win beats need, fold if it does not

Both numbers come from tables measured on the arena's public replays (`build_price_tables.py` writes
`price_tables.json`):

- `value`: what a stack is worth in rating points, by players left, stage of the game and stack size. Folding keeps the
  stack, calling and losing leaves the smaller one (busting is worth exactly the place of the last player left),
  calling and winning gives the bigger one. `need = (fold - lose) / (win - lose)`. Early, with six players left,
  a call for the whole stack needs about 70%+; with three left it needs far less than the chip price.
- `preflop` / `postflop`: the average real equity of each kind of hand in each kind of spot.

`card(messages)` reads the official prompt; nothing here uses hidden cards, so the same card can be made for any
decision the model is shown. The text from `hint(card)` is for the OPSD teacher only.
"""

from __future__ import annotations

import json
import re
from bisect import bisect_right
from functools import cache
from pathlib import Path
from typing import Any

TABLES = Path(__file__).with_name("price_tables.json")
PLACE_POINTS = {1: 0.83, 2: 0.63, 3: 0.38, 4: 0.05, 5: -0.45, 6: -1.45}  # rating change by finishing place, six seats
RANKS = "23456789TJQKA"
CLOSE = 0.05  # win and need closer than this: no clear answer

_HAND = re.compile(r"## Hand (\d+) of (\d+)")
_STATE = re.compile(
    r"Street: (?P<street>preflop|flop|turn|river)\nYour hole cards: (?P<hole>\S+ \S+)\nBoard: (?P<board>[^\n]*)\n"
    r"Pot \(including current bets\): (?P<pot>\d+)\nTable:\n(?P<seats>(?:  seat [^\n]+\n)+)"
)
_SEAT = re.compile(r"  seat (\d+): (.+?) stack (\d+), bet (\d+)(?:, (folded|all-in|out))?$")
_CALL = re.compile(r"Legal actions: fold, call (\d+)")

PREFLOP_NAMES = {
    "QQ+": "a premium pair (queens or better)", "99-JJ": "a medium pair (nines to jacks)", "22-88": "a small pair",
    "AK": "ace-king", "AQ-AJ": "ace-queen or ace-jack", "Ax": "an ace with a weaker kicker",
    "broadway": "two cards ten or higher without an ace", "suited": "other suited cards", "offsuit": "other offsuit cards",
}
POSTFLOP_NAMES = {
    "monster": "a straight or better", "trips": "three of a kind", "two_pair": "two pair", "overpair": "an overpair",
    "top_pair": "top pair", "low_pair": "a pair below the top card", "draw": "a flush or open-ended straight draw and no pair",
    "nothing": "no pair and no strong draw",
}


def preflop_class(hole: list[str]) -> str:
    a, b = sorted((RANKS.index(c[0]) for c in hole), reverse=True)
    if a == b:
        return "QQ+" if a >= 10 else "99-JJ" if a >= 7 else "22-88"
    if a == 12:
        return "AK" if b == 11 else "AQ-AJ" if b >= 9 else "Ax"
    if b >= 8:
        return "broadway"
    return "suited" if hole[0][1] == hole[1][1] else "offsuit"


def _straight(ranks: set[int]) -> bool:
    ranks = ranks | ({-1} if 12 in ranks else set())
    return any(all(r + i in ranks for i in range(5)) for r in range(-1, 9))


def _strength(cards: list[str]) -> int:
    """0 high card, 1 pair, 2 two pair, 3 trips, 4 straight, 5 flush, 6 full house or better."""
    counts = sorted((sum(c[0] == r for c in cards) for r in {c[0] for c in cards}), reverse=True)
    flush = any(sum(c[1] == s for c in cards) >= 5 for s in "cdhs")
    straight = _straight({RANKS.index(c[0]) for c in cards})
    if counts[0] >= 4 or (counts[0] == 3 and len(counts) > 1 and counts[1] >= 2):
        return 6
    return 5 if flush else 4 if straight else 3 if counts[0] == 3 else 2 if counts[:2] == [2, 2] else 1 if counts[0] == 2 else 0


def postflop_class(hole: list[str], board: list[str]) -> str:
    everything = hole + board
    made = _strength(everything)
    if made >= 4 and made > _strength(board):
        return "monster"
    count = {r: sum(c[0] == r for c in everything) for r in {c[0] for c in hole}}
    top = max(RANKS.index(c[0]) for c in board)
    if any(n >= 3 for n in count.values()):
        return "trips"
    pocket = hole[0][0] == hole[1][0]
    paired = [r for r, n in count.items() if n >= 2]
    if not pocket and len(paired) == 2:
        return "two_pair"
    if pocket:
        return "overpair" if RANKS.index(hole[0][0]) > top else "low_pair"
    if paired:
        return "top_pair" if RANKS.index(paired[0]) == top else "low_pair"
    if len(board) < 5:
        flush_draw = any(sum(c[1] == s for c in everything) == 4 and any(c[1] == s for c in hole) for s in "cdhs")
        ranks = {RANKS.index(c[0]) for c in everything} | ({-1} if any(c[0] == "A" for c in everything) else set())
        open_ended = any(all(r + i in ranks for i in range(4)) for r in range(9))
        if flush_draw or open_ended:
            return "draw"
    return "nothing"


def parse_state(messages: list[Any]) -> tuple[dict[str, Any], str] | None:
    """Read the latest public table state and return the decision text that follows it."""
    users = [m for m in messages if _role(m) == "user" and isinstance(_content(m), str)]
    if not users:
        return None
    last = _content(users[-1])
    state = None
    for state in _STATE.finditer(last):
        pass  # the last state block in the message
    hands = [h for m in users for h in _HAND.finditer(_content(m))]
    if not state or not hands:
        return None
    seats = []
    for line in state["seats"].splitlines():
        seat = _SEAT.match(line)
        if not seat:
            return None
        seats.append({
            "seat": int(seat[1]), "you": "(you)" in seat[2], "button": "[button]" in seat[2],
            "stack": int(seat[3]), "bet": int(seat[4]), "status": seat[5] or "",
        })
    return {
        "hand": int(hands[-1][1]), "hands": int(hands[-1][2]), "street": state["street"], "hole": state["hole"].split(),
        "board": [] if state["board"].strip() in ("-", "") else state["board"].split(), "pot": int(state["pot"]),
        "live": sum(s["status"] != "out" for s in seats), "seats": seats,
    }, last[state.end():]


def parse(messages: list[Any]) -> dict[str, Any] | None:
    """The spot a decision prompt describes, or None when it is not a fold-or-call decision."""
    parsed = parse_state(messages)
    if parsed is None:
        return None
    spot, rest = parsed
    call = _CALL.search(rest)
    if call is None:
        return None
    seats = spot.pop("seats")
    me = [s for s in seats if s["you"]]
    if len(me) != 1:
        return None
    me = me[0]
    to_call = int(call[1])
    cap = me["bet"] + min(to_call, me["stack"])
    others = [s for s in seats if not s["you"] and s["status"] not in ("folded", "out")]
    committed = [s for s in others if s["bet"] >= me["bet"] + to_call or s["status"] == "all-in"]
    all_in = [s for s in committed if s["status"] == "all-in"]
    return spot | {
        "to_call": to_call, "stack": me["stack"],
        "committed": max(1, len(committed)), "all_in": len(all_in), "all_in_size": max((s["bet"] for s in all_in), default=0),
        "street_bet": me["bet"] + to_call,
        "covered_all_ins": sum(s["bet"] <= cap for s in all_in),
        "unreachable_street_bets": sum(max(0, s["bet"] - cap) for s in seats if not s["you"]),
    }


def _role(m: Any) -> str:
    return m.get("role", "") if isinstance(m, dict) else getattr(m, "role", "")


def _content(m: Any) -> Any:
    return m.get("content") if isinstance(m, dict) else getattr(m, "content", None)


def spot_keys(spot: dict[str, Any]) -> list[str]:
    """Table keys for the spot's win rate, most specific first."""
    many = "multi" if spot["committed"] >= 2 else "one"
    if spot["street"] == "preflop":
        kind = preflop_class(spot["hole"])
        bet = "raise" if not spot["all_in"] else "shove_short" if spot["all_in_size"] < 100 else "shove"
        return [f"pre|{kind}|{bet}|{many}", f"pre|{kind}|{bet}", f"pre|{kind}"]
    kind = postflop_class(spot["hole"], spot["board"])
    price = spot["to_call"] / (spot["pot"] + spot["to_call"])
    size = "small" if price < 0.25 else "big" if price < 0.36 else "huge"
    when = "river" if spot["street"] == "river" else "early"
    return [f"post|{kind}|{when}|{size}", f"post|{kind}|{when}", f"post|{kind}"]


@cache
def tables() -> dict[str, Any]:
    return json.loads(TABLES.read_text())


def stack_value(live: int, hand: int, stack: int, table: dict[str, Any] | None = None) -> float:
    """Rating points a stack is worth with `live` players left at this stage of the game."""
    live = max(1, min(6, live))
    if stack <= 0:
        return PLACE_POINTS[live]
    if live == 1:
        return PLACE_POINTS[1]
    points = (table or tables()["value"])[f"{live}|{0 if hand <= 10 else 1 if hand <= 20 else 2}"]
    xs = [0] + [p[0] for p in points]
    ys = [PLACE_POINTS[live]] + [p[1] for p in points]
    if stack >= xs[-1]:
        return ys[-1]
    i = bisect_right(xs, stack)
    return ys[i - 1] + (ys[i] - ys[i - 1]) * (stack - xs[i - 1]) / (xs[i] - xs[i - 1])


def win_rate(spot: dict[str, Any], table: dict[str, Any] | None = None) -> float:
    """Shrunk average equity of this kind of hand in this kind of spot."""
    table = table or tables()["win"]
    estimate = None
    for key in reversed(spot_keys(spot)):  # widest first; each narrower cell pulls the estimate toward itself
        cell = table.get(key)
        if not cell:
            continue
        mean, n = cell
        estimate = mean if estimate is None else (mean * n + estimate * 12) / (n + 12)
    return 0.4 if estimate is None else estimate


def card(messages: list[Any], table: dict[str, Any] | None = None) -> dict[str, Any] | None:
    spot = parse(messages)
    if spot is None or spot["to_call"] <= 0:
        return None
    table = table or tables()
    stack, to_call, pot, live, hand = spot["stack"], min(spot["to_call"], spot["stack"]), spot["pot"], spot["live"], spot["hand"]
    # These public street bets are a lower bound on chips the caller cannot win. Earlier multiway side pots
    # need the full commitment ledger, which the terminal call teacher uses for its narrower class of spots.
    pot -= spot["unreachable_street_bets"]
    busted = spot["covered_all_ins"]
    fold = stack_value(live, hand, stack, table["value"])
    lose = stack_value(live, hand, stack - to_call, table["value"])
    won = stack_value(live - busted, hand, stack + pot, table["value"])
    price = to_call / (pot + to_call)
    need = (fold - lose) / (won - lose) if won - lose > 0.02 else price
    need = min(0.95, max(0.03, need))
    win = win_rate(spot, table["win"])
    if spot["street"] == "preflop":
        kind = PREFLOP_NAMES[preflop_class(spot["hole"])]
    else:
        made = postflop_class(spot["hole"], spot["board"])
        kind = POSTFLOP_NAMES[made]
        if made == "monster":  # name the hand exactly: the model misreads its own strong hands
            kind = {4: "a straight", 5: "a flush"}.get(_strength(spot["hole"] + spot["board"]), "a full house or better")
    return spot | {
        "price": price, "need": need, "win": win, "fold_value": fold, "lose_value": lose, "win_value": won,
        "share": to_call / max(1, stack), "kind": kind, "contestable_pot": pot,
        "plan": "close" if abs(win - need) < CLOSE else "continue" if win > need else "fold",
    }


def hint(c: dict[str, Any]) -> str:
    """Teacher-only text: the numbers and the order to reason in."""
    stack, to_call, pot, live = c["stack"], min(c["to_call"], c["stack"]), c["contestable_pot"], c["live"]
    bust = stack - to_call <= 0
    after_loss = (f"you are out in place {live} of 6, the worst result still possible" if bust
                  else f"you are left with {stack - to_call} chips")
    facing = ("an all-in" if c["all_in"] else "a raise") if c["street"] == "preflop" else f"a bet on the {c['street']}"
    callers = " with more than one opponent already in" if c["committed"] >= 2 else ""
    direction = "more" if c["need"] > c["price"] + 0.03 else "less" if c["need"] < c["price"] - 0.03 else "same"
    why = {
        "more": "because losing these chips costs more places than winning them gains",
        "less": "because enough players are already out that losing costs little and winning moves you up",
        "same": "because here chips and places move together",
    }[direction]
    versus = {"more": "more than", "less": "less than", "same": "about the same as"}[direction]
    verdict = {
        "fold": f"{c['win']:.0%} is short of {c['need']:.0%}: fold.",
        "continue": f"{c['win']:.0%} beats {c['need']:.0%}: continue (call, or raise if the hand is strong enough to want more chips in).",
        "close": f"{c['win']:.0%} against {c['need']:.0%} is close: use what you know about this opponent to break the tie.",
    }[c["plan"]]
    return (
        "Your read of this decision. These are your own estimates from experience at these tables: in your reasoning, "
        "state them as your own thinking, and do not mention notes, a prompt, instructions, or being given anything.\n"
        f"- Chip price: calling {to_call} to win {pot} needs {c['price']:.0%}.\n"
        f"- Place price: you are ranked by finishing place, with {live} players left on hand {c['hand']} of {c['hands']}. "
        f"If you fold you keep {stack} chips. If you call and lose, {after_loss}. If you call and win you have "
        f"{stack + pot}. Weighing those three outcomes by what they do to your finishing place, the call has to win "
        f"about {c['need']:.0%} of the time, {versus} the chip price, {why}.\n"
        f"- Your hand: {c['kind']}, facing {facing}{callers}. In this field, hands like this have won about "
        f"{c['win']:.0%} of the time in this kind of spot.\n"
        f"- Compare: {verdict}\n"
        "Think in that order and keep it short: the chip price, what losing and winning do to your finishing place, "
        "how often a hand like yours wins against the hands that bet this way, then the comparison. Act on the "
        "comparison even when general caution or the size of the pot pulls the other way, then call submit_action "
        "exactly once."
    )
