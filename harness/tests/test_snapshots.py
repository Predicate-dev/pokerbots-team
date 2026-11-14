"""Snapshots cut from a finished game replay each hand exactly as the game dealt it, with the ledger a perfect
note-taker would have had."""

import json
import sys
import zlib
from pathlib import Path

import pytest
from test_official_rules import OFFICIAL, bot, last_user, seats, submit

from poker_harness.completions import rules_output, table_rules, with_preset
from poker_harness.game import play_session
from poker_harness.ledger import from_params, parse, render, to_params, update
from poker_harness.prompts import TableRules
from poker_harness.snapshots import snapshots

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cut_snapshots

GAME = TableRules(**{**OFFICIAL.__dict__, "seats": 4, "play_hands": 10})  # the first 10 hands of a 30-hand game


def moody(messages: list[dict]):
    """A fixed mix of folds, calls and small bets that depends only on what the seat is shown."""
    text = last_user(messages)
    mood = zlib.crc32(text.encode())
    legal = text.split("Legal actions:")[-1]
    if "check" in legal:
        return submit("bet", 4) if mood % 3 == 0 and "bet to" in legal else submit("check")
    return submit("fold") if mood % 4 == 0 else submit("call")


def table(rules: TableRules) -> list:
    return seats(4, rules, **{f"seat{i}": bot(moody) for i in range(4)})


def moves(hand: dict) -> list[tuple]:
    return [(a["player"], a["street"], a["action"]) for a in hand["actions"]]


async def game() -> dict:
    return await play_session("g", 21, table(GAME), GAME)


async def test_a_snapshot_replays_the_hand_exactly_as_the_game_dealt_it() -> None:
    log = await game()
    assert len(log["hands"]) == 10 and len({h["button_seat"] for h in log["hands"]}) > 1
    spots = snapshots(log)
    for spot in (s for s in spots if s["first_hand"] in (2, 5, 9) and s["policy_seats"] == [2]):
        original = log["hands"][spot["first_hand"] - 1]
        start = {k: spot[k] for k in ("first_hand", "play_hands", "button_seat")}
        rules = TableRules(
            **{**OFFICIAL.__dict__, "seats": 4, "starting_stacks": tuple(spot["starting_stacks"]), **start}
        )
        replay = (await play_session("g2", spot["seed"], table(rules), rules))["hands"][0]
        assert replay["hand_number"] == original["hand_number"] and replay["button_seat"] == original["button_seat"]
        assert replay["starting_state"] == original["starting_state"]  # the same cards and stacks in the same seats
        assert moves(replay) == moves(original) and replay["ending_stacks"] == original["ending_stacks"]
        assert replay["final_board"] == original["final_board"]


async def test_each_snapshot_carries_the_ledger_of_the_hands_before_it() -> None:
    log = await game()
    spots = snapshots(log, policy=["seat1"])
    assert [s["first_hand"] for s in spots] == list(range(1, 11)) and all(s["policy_seats"] == [1] for s in spots)
    assert "starting_ledger" not in spots[0] and spots[0]["hands_left"] == 29  # a game starts with blank notes
    ledger = None
    for spot, hand in zip(spots, log["hands"], strict=True):
        assert spot["starting_stacks"] == [hand["starting_state"][f"seat{i}"]["starting_stack"] for i in range(4)]
        assert spot.get("starting_ledger") == (to_params(ledger) if ledger else None)
        assert spot["hands_left"] == 30 - hand["hand_number"] and spot["seed"] == 21
        ledger = update(ledger, hand, "seat1", 30)
    last = from_params(spots[-1]["starting_ledger"])
    assert (last.hand, last.total, last.me) == (9, 30, "seat1")
    assert all(seat.counts["n"] == 9 for seat in last.seats.values())
    assert sum(seat.counts["vp"] for seat in last.seats.values()) > 0 and parse(render(last)) == last
    stacks = [seat.stack for seat in last.seats.values()]
    assert stacks == spots[-1]["starting_stacks"]  # the ledger's stacks are the table's


async def test_filters_and_seats_that_are_out() -> None:
    log = await game()
    assert len(snapshots(log)) == 40 and len(snapshots(log, min_hand=8)) == 12
    assert {s["policy_seats"][0] for s in snapshots(log, policy=["seat0", "seat3"])} == {0, 3}
    # Bust seat2 in the log before hand 6: it gets no more snapshots and shows as out in everyone else's.
    for hand in log["hands"][5:]:
        del hand["starting_state"]["seat2"]
        hand["ending_stacks"]["seat2"] = 0
    spots = snapshots(log, min_hand=7)
    assert {s["policy_seats"][0] for s in spots} == {0, 1, 3} and all(s["starting_stacks"][2] == 0 for s in spots)
    assert all(s["starting_ledger"]["seats"]["seat2"] == {"stack": 0} for s in spots)
    for bad in ({"policy": ["seat9"]},):
        with pytest.raises(ValueError):
            snapshots(log, **bad)
    with pytest.raises(ValueError):
        snapshots(log | {"session_info": log["session_info"] | {"player_names": ["a", "b", "c", "d"]}})
    with pytest.raises(ValueError):
        snapshots(log | {"session_info": {k: v for k, v in log["session_info"].items() if k != "seed"}})


def test_the_button_setting_reaches_the_rules_and_event_tables_are_unchanged() -> None:
    assert table_rules(with_preset({"preset": "event"})).button_seat is None
    rules = table_rules(with_preset({"preset": "event", "button_seat": 0, "first_hand": 4}))
    assert rules.button_seat == 0 and rules_output(rules)["button_seat"] == 0
    assert "button_seat" not in rules_output(table_rules(with_preset({"preset": "event"})))


async def test_a_button_seat_that_is_not_at_the_table_is_refused() -> None:
    rules = TableRules(**{**OFFICIAL.__dict__, "seats": 4, "button_seat": 7})
    with pytest.raises(ValueError):
        await play_session("g", 1, table(rules), rules)


async def test_the_command_line_tool_writes_one_line_per_snapshot_and_skips_what_it_cannot_use(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log = await game()
    (tmp_path / "game_a.json").write_text(json.dumps(log | {"game_id": None}))
    (tmp_path / "one_hand.json").write_text(json.dumps(log | {"hands": log["hands"][:1]}))  # a training rollout
    (tmp_path / "not_a_log.json").write_text(json.dumps({"hands": [{}] * 9}))
    out = tmp_path / "out" / "snapshots.jsonl"
    cut_snapshots.main([*map(str, sorted(tmp_path.glob("*.json"))), "--out", str(out), "--policy", "seat0"])
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    # A log without an id is named after its file, so its hands can be kept together when datasets are split.
    assert len(lines) == 10 and lines == [s | {"game_id": "game_a"} for s in snapshots(log, policy=["seat0"])]
    printed = capsys.readouterr().out
    assert '"games": 1' in printed and '"skipped: too few hands": 1' in printed and '"skipped: KeyError": 1' in printed
    assert "wrote 10 snapshots from 1 games" in printed

    cut_snapshots.main([str(tmp_path / "one_hand.json"), "--out", str(tmp_path / "none.jsonl")])
    assert not (tmp_path / "none.jsonl").exists() and "no snapshots" in capsys.readouterr().out
