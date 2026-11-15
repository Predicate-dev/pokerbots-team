import json
from argparse import Namespace

from pokerkit import HandHistory

from build_pluribus_puzzles import hand_puzzles
from poker_harness.seat import Completion
from puzzles import run


def recorded_hand(actions: list[str]) -> HandHistory:
    return HandHistory(
        variant="NT",
        antes=[0] * 6,
        blinds_or_straddles=[50, 100, 0, 0, 0, 0],
        min_bet=100,
        starting_stacks=[10_000] * 6,
        players=["Alice", "Pluribus", "Chris", "Dana", "Eli", "Fin"],
        actions=[
            "d dh p1 2c3h",
            "d dh p2 4d5s",
            "d dh p3 6s7d",
            "d dh p4 8h9s",
            "d dh p5 AhKd",
            "d dh p6 QsJd",
            *actions,
        ],
    )


async def test_river_puzzles_preserve_fractional_chips_and_score_with_existing_runner(tmp_path, monkeypatch, capsys):
    hand = recorded_hand(
        [
            "p3 f",
            "p4 f",
            "p5 f",
            "p6 cbr 213",
            "p1 cc",
            "p2 cc",
            "d db AcKcQc",
            "p1 cc",
            "p2 cc",
            "p6 cc",
            "d db Jc",
            "p1 cc",
            "p2 cc",
            "p6 cc",
            "d db Tc",
            "p1 cbr 385",
            "p2 cc",
            "p6 cc",
        ]
    )
    all_rows = list(hand_puzzles(hand, "sample/0"))
    rows = list(hand_puzzles(hand, "sample/0", "Pluribus"))
    assert len(all_rows) == 2
    assert rows == all_rows[:1]
    row = rows[0]
    assert row["kind"] == "river_bet"
    assert (row["to_call"], row["pot"], row["equity"]) == (7.7, 20.48, 0.333)
    assert row["took"] == row["best"] == "continue"
    assert row["villains"] == 2
    assert not row["closes_action"]
    prompt = row["messages"][-1]["content"]
    assert "Your hole cards: 4d 5s" in prompt
    assert "raise to 4.26" in prompt
    assert "seat0 (seat 0) bet to 7.7" in prompt
    assert "river: seat1" not in prompt
    assert "Pluribus" not in prompt
    assert "2c" not in prompt and "Qs" not in prompt

    async def complete(messages):
        assert messages == row["messages"]
        return Completion(text="", tool_arguments={"action": "call"})

    file = tmp_path / "puzzles.jsonl"
    file.write_text(json.dumps(row) + "\n")
    monkeypatch.setenv("AC2_API_KEY", "test-key")
    monkeypatch.setattr("puzzles.relay_completion", lambda *args: complete)
    await run(Namespace(puzzles=file, model="test", parallel=1, only_ours=False, limit=None, out=None))
    assert "1/1 best" in capsys.readouterr().out


def test_large_call_uses_stack_before_the_recorded_action():
    actions = ["p3 cbr 4000", "p4 f", "p5 f", "p6 f", "p1 f", "p2 f"]
    hand = recorded_hand(actions)
    rows = list(hand_puzzles(hand, "sample/0", "Fin"))
    assert len(rows) == 1
    assert (rows[0]["kind"], rows[0]["to_call"], rows[0]["recorded_action"]) == ("stack_call", 80, "fold")
    actions[0] = "p3 cbr 3999"
    assert not list(hand_puzzles(recorded_hand(actions), "sample/0", "Fin"))
