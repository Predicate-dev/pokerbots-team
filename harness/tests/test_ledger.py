"""The seat ledger: its layout round-trips, the reference update counts a real hand correctly, and `accuracy`
gives part marks."""

import pytest
from test_official_rules import OFFICIAL, bot, last_user, seats, submit

from poker_harness.game import play_session
from poker_harness.ledger import COUNTERS, GOAL, KEY, Ledger, SeatLine, accuracy, from_params, parse, render, update
from poker_harness.prompts import TableRules

RULES = TableRules(**{**OFFICIAL.__dict__, "seats": 4, "total_hands": 30, "play_hands": 1, "first_hand": 7})
PARAMS = {
    "hand": 26,
    "total": 30,
    "me": "seat3",
    "seats": {
        "seat0": {
            "stack": 155,
            "n": 26,
            "vp": 9,
            "pr": 6,
            "3b": 1,
            "ab": 7,
            "pc": 4,
            "fd": 5,
            "sd": 2,
            "open": [5, 5, 6],
        },
        "seat1": {"stack": 275, "n": 26, "vp": 14, "pr": 3, "pc": 11, "note": "calls too much"},
        "seat2": {"stack": 0},
        "seat3": {"stack": 218, "n": 26, "vp": 8, "pr": 7, "open": [4, 5, 5, 5]},
    },
    "notes": ["seat1 said nice hand twice"],
}


def test_the_layout_round_trips_and_fits_the_scratchpad() -> None:
    ledger = from_params(PARAMS)
    text = render(ledger)
    lines = text.splitlines()
    assert lines[0] == "LEDGER v1 | after hand 26 of 30 | me seat3" and lines[1:3] == [GOAL, KEY]
    assert "seat0 stack 155 | n26 vp9 pr6 3b1 ab7 pc4 fd5 sd2 | open 5,5,6 ||" in lines
    assert "seat1 stack 275 | n26 vp14 pr3 3b0 ab0 pc11 fd0 sd0 | open - || calls too much" in lines
    assert "seat2 out" in lines and lines[-1] == "seat1 said nice hand twice"
    assert "seat3 stack 218 (me) | n26 vp8 pr7 3b0 ab0 pc0 fd0 sd0 | open 5,5,5 ||" in lines  # the last three opens
    assert parse(text) == ledger and render(parse(text)) == text
    six = {f"seat{i}": PARAMS["seats"]["seat0"] | {"note": "x" * 120} for i in range(6)}
    assert len(render(from_params(PARAMS | {"seats": six}))) < 2500  # half the 5,000 left for the model's notes


def test_parse_is_forgiving_about_everything_but_the_facts() -> None:
    assert parse("") is None and parse("seat0 looks tight") is None
    messy = "my notes first\n" + render(from_params(PARAMS)).replace("vp9 pr6", "vp9  pr6") + "\nseat9 stack lots\n"
    ledger = parse(messy)
    assert ledger.seats["seat0"].counts["pr"] == 6 and "seat9" not in ledger.seats
    assert ledger.notes[-1] == "seat9 stack lots"  # a line that doesn't fit is kept as a note, not lost
    assert ledger.seats["seat2"].stack == 0 and ledger.seats["seat1"].note == "calls too much"


async def played_hand() -> dict:
    """seat2 opens to 6, seat3 reraises to 18, the blinds fold, seat2 calls; seat2 check-folds the flop."""

    def seat2(messages: list[dict]):
        text = last_user(messages)
        if "Street: preflop" in text:
            return submit("call") if "seat3 (seat 3) raise" in text else submit("raise", 6)
        return submit("fold") if "call " in text.split("Legal actions:")[-1] else submit("check")

    def seat3(messages: list[dict]):
        return submit("raise", 18) if "Street: preflop" in last_user(messages) else submit("bet", 20)

    fold = bot(lambda m: submit("fold"))
    table = seats(4, RULES, seat0=fold, seat1=fold, seat2=bot(seat2), seat3=bot(seat3))
    return (await play_session("g", 5, table, RULES))["hands"][0]


async def test_update_counts_a_real_hand_from_nothing() -> None:
    hand = await played_hand()
    kinds = [(a["player"], a["street"], a["action"]["type"]) for a in hand["actions"]]
    assert ("seat3", "preflop", "raise") in kinds and ("seat2", "flop", "fold") in kinds and not hand["showdown"]
    ledger = update(None, hand, "seat0", 30)
    assert (ledger.hand, ledger.total, ledger.me) == (7, 30, "seat0")
    counts = {name: seat.counts for name, seat in ledger.seats.items()}
    zero = dict.fromkeys(COUNTERS, 0)
    assert counts["seat0"] == counts["seat1"] == zero | {"n": 1}  # posting a blind and folding is not paying to play
    assert counts["seat2"] == zero | {"n": 1, "vp": 1, "pr": 1, "fd": 1}
    assert counts["seat3"] == zero | {"n": 1, "vp": 1, "pr": 1, "3b": 1, "ab": 1}
    assert ledger.seats["seat2"].opens == [6] and ledger.seats["seat3"].opens == []  # a reraise is not an open
    assert {name: seat.stack for name, seat in ledger.seats.items()} == hand["ending_stacks"]


async def test_update_adds_to_the_ledger_it_was_given_and_keeps_the_notes() -> None:
    hand = await played_hand()
    prior = from_params(PARAMS | {"hand": 6, "me": "seat3"})
    after = update(prior, hand, "seat3", 30)
    assert after.hand == 7 and after.notes == ["seat1 said nice hand twice"]
    assert after.seats["seat1"].counts == prior.seats["seat1"].counts | {"n": 27}
    assert after.seats["seat1"].note == "calls too much"
    assert after.seats["seat3"].counts["3b"] == 1 and after.seats["seat3"].counts["pr"] == 8
    assert after.seats["seat2"].counts["n"] == 1  # it was "out" in the notes but got dealt in: counted from zero
    assert update(prior, hand, "seat0", 30).seats["seat1"].counts["n"] == 1  # someone else's ledger is not carried


async def test_a_showdown_counts_for_every_seat_still_in() -> None:
    table = seats(4, RULES)  # everyone calls to the river
    hand = (await play_session("g", 5, table, RULES))["hands"][0]
    ledger = update(None, hand, "seat0", 30)
    assert hand["showdown"] and all(seat.counts["sd"] == 1 for seat in ledger.seats.values())
    assert ledger.seats["seat2"].counts["vp"] == 1 and ledger.seats["seat1"].counts["vp"] == 0  # the big blind checked


async def test_accuracy_gives_part_marks_and_zero_without_a_ledger() -> None:
    reference = update(None, await played_hand(), "seat0", 30)
    assert accuracy(render(reference), reference) == 1.0
    assert accuracy("seat2 opened to 6 and folded the flop", reference) == 0.0
    facts = 1 + 4 * (2 + len(COUNTERS))
    one_off = render(reference).replace("vp1 pr1 3b1", "vp1 pr1 3b0")
    assert accuracy(one_off, reference) == pytest.approx((facts - 1) / facts)
    missing_a_seat = "\n".join(line for line in render(reference).splitlines() if not line.startswith("seat1 "))
    assert accuracy(missing_a_seat, reference) == pytest.approx((facts - 10) / facts)
    with_notes = render(Ledger(7, 30, "seat0", reference.seats, ["seat3 reraises a lot"]))
    assert accuracy(with_notes, reference) == 1.0  # the model's own lines are never graded
    out = Ledger(7, 30, "seat0", {"seat0": SeatLine(400), "seat1": SeatLine(0)})
    assert accuracy(render(out), out) == 1.0 and accuracy(render(out).replace("seat1 out", ""), out) < 1.0
