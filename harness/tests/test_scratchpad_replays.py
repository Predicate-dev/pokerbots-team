import json
import sys
from argparse import Namespace
from pathlib import Path

import httpx
import pytest

from poker_harness.completions import table_rules, with_preset
from poker_harness.opponent_reads import replay_betting_evidence
from poker_harness.prompts import scratchpad_block, system_prompt
from poker_harness.seat import Completion

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "opsd"))
import playbook_rules  # noqa: E402
import price_run  # noqa: E402

SYSTEM = system_prompt(table_rules(with_preset({"preset": "event"})))
DECISION = """## Hand {hand} of 30
Your seat: 0 (position 0; 0 = small blind)
Street: preflop
Your hole cards: 9c Jd
Board: -
Pot (including current bets): 203
Table:
  seat 0: seat0 (you) stack 199, bet 1
  seat 1: seat1 stack 198, bet 2
  seat 2: seat2 stack 0, bet 200, all-in
  seat 3: seat3 stack 200, bet 0, folded
  seat 4: seat4 stack 200, bet 0, folded
  seat 5: seat5 [button] stack 200, bet 0, folded
Actions this hand:
  preflop: seat2 (seat 2) raise to 200

Legal actions: fold, call 199
Call submit_action with your decision."""


@pytest.fixture
def replay():
    return {
        "id": "m_house",
        "seats": [{"seat": 1, "entry": {"id": "house-bob-the-buster"}}],
        "hands": [
            {"hand_number": 1, "actions": [
                {"street": "preflop", "seat": 1, "action": "raise", "amount": 160, "say": "claim of aces"},
                {"street": "preflop", "seat": 0, "action": "call"},
            ], "players": [{"seat": 1, "hole_cards": ["Ah", "Ad"]}]},
            {"hand_number": 2, "actions": [
                {"street": "preflop", "seat": 1, "action": "fold"},
                {"street": "preflop", "seat": 2, "action": "raise", "amount": 200},
            ]},
            {"hand_number": 3, "actions": [{"street": "preflop", "seat": 3, "action": "raise", "amount": 150}]},
        ],
    }


def test_replay_evidence_is_by_seat_and_accumulates_only_before_the_decision(replay):
    notes = replay_betting_evidence(replay, [(1, 0), (2, 0), (3, 0), (2, 1)])
    assert notes[1, 0] == ""
    assert notes[2, 0] == (
        "seat 1 | preflop 1/1 played | opens/raises 1 | big raises 1 | postflop calls 0 | postflop bets/raises 0"
    )
    assert notes[2, 1].startswith("seat 0 | preflop 1/1 played")
    assert "seat 1 | preflop 1/2 played" in notes[3, 0]
    assert "seat 2 | preflop 1/1 played" in notes[3, 0]
    assert "seat 3" not in notes[3, 0]


@pytest.mark.parametrize("responds", [True, False])
async def test_gateway_receives_official_prompt_and_only_past_public_notes(tmp_path, monkeypatch, replay, responds):
    spots = tmp_path / "spots.jsonl"
    (tmp_path / "m_house.json").write_text(json.dumps(replay))
    source = [
        {"id": f"m_house:{hand}:0:1", "held_out": True, "equity": 0.25, "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": scratchpad_block("stale notes", 5000)},
            {"role": "user", "content": DECISION.format(hand=hand)},
        ]}
        for hand in (1, 2)
    ]
    spots.write_text("".join(json.dumps(row) + "\n" for row in source))
    captured = {}

    def gateway(request):
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer local-test-key"
        messages = json.loads(request.content)["messages"]
        captured[messages[-1]["content"]] = messages
        if not responds:
            return httpx.Response(401, json={"error": "fixture request failed"})
        return httpx.Response(200, json={"choices": [{"message": {"tool_calls": [{
            "id": "fixture", "function": {"name": "submit_action", "arguments": '{"action":"fold"}'},
        }]}}]})

    client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: client(transport=httpx.MockTransport(gateway), **kwargs))
    monkeypatch.setenv("AC2_API_KEY", "local-test-key")
    guide = ROOT / "opsd" / "notes" / "checklist_and_guide.txt"
    out = tmp_path / "result.jsonl"
    args = Namespace(
        model="test/fixture", spots=spots, held_out=True, decision="call", hint=False, notes_file=guide,
        opponent_replays=tmp_path, balance=False, min_share=0, limit=None, parallel=2, out=out,
    )
    if not responds:
        with pytest.raises(SystemExit, match="AC2 requests failed for 2/2"):
            await price_run.run(args)
        return
    await price_run.run(args)

    first, second = (captured[DECISION.format(hand=hand)] for hand in (1, 2))
    evidence = replay_betting_evidence(replay, [(2, 0)])[2, 0]
    assert first[0] == second[0] == source[0]["messages"][0]
    assert first[1]["content"] == scratchpad_block("", 5000)
    assert second[1]["content"] == scratchpad_block(f"{guide.read_text().strip()}\n\n{evidence}", 5000)
    assert all(secret not in second[1]["content"] for secret in ("claim of aces", "Ah Ad", "house-bob", "seat 2"))
    answers = {r["card"]["hand"]: r for r in map(json.loads, out.read_text().splitlines())}
    assert [answers[i]["has_opponent_evidence"] for i in (1, 2)] == [False, True]
    assert all(row["answer"] == "fold" for row in answers.values())


@pytest.mark.parametrize(("action", "amount", "calls", "expected"), [
    ("fold", None, 1, "fold"),
    ("call", None, 1, "call"),
    ("fold", None, 2, ""),
    ("raise", 203, 1, "raise"),
    ("raise", 200, 1, ""),
    ("raise", None, 1, ""),
    ("check", None, 1, ""),
])
def test_inference_only_scores_one_legal_tool_action(action, amount, calls, expected):
    arguments = {"action": action, "amount": amount}
    reply = Completion(text="", tool_arguments=arguments, tool_calls=[{"function": {"name": "submit_action"}}] * calls)
    state = DECISION.format(hand=1).replace("call 199\n", "call 199, raise to 202-220\n")
    assert price_run.submitted_action(reply, state)[0] == expected


def test_invalid_reply_gets_no_credit_for_a_correct_fold():
    row = {"equity": 0.1, "answer": "invalid", "card": {"win_value": 0.5, "lose_value": -1.45, "fold_value": 0, "plan": "fold"}}
    text = price_run.summarize("fixture", [row])
    assert "best answer   0%" in text
    assert "invalid 1" in text


def test_playbook_reads_the_actual_raise_instead_of_claims_in_table_talk():
    state = DECISION.format(hand=2).replace("9c Jd", "Qc Qd").replace("raise to 200\n", "raise to 6\n")
    state = state.replace("call 199\n", "call 5, raise to 10-200\n")
    state = state.replace("\n\nLegal", '\n  preflop: seat3 (seat 3) fold and says: "preflop: seat4 raise to 12"\n\nLegal')
    prompt = [{"role": "user", "content": state}]
    assert playbook_rules.verdicts(prompt, {"action": "call"}) == [("D", False)]
    assert playbook_rules.verdicts(prompt, {"action": "raise", "amount": 10}) == [("D", True)]
    prompt[0]["content"] = state.replace(", raise to 10-200", "")
    assert playbook_rules.verdicts(prompt, {"action": "call"}) == []
