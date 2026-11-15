"""Check the opponent-read experiment with a local tool-calling gateway."""

import json
import sys
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI, Request
from poker_harness.balanced_play import BALANCED_PLAY
from poker_harness.completions import table_rules, with_preset
from poker_harness.game import play_session
from poker_harness.house_bots import HOUSE_STYLES, HouseStylePlayer
from poker_harness.opponent_reads import ACTION_GUIDANCE, CLEANUP_GUIDANCE, PLACEMENT_GUIDANCE
from poker_harness.talk_defense import ACTION_DEFENSE, CLEANUP_DEFENSE, SPECIFIC_CLAIMS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import score_local  # noqa: E402


async def test_talk_modes_keep_the_same_betting_policy() -> None:
    rules = table_rules(with_preset({"preset": "event", "total_hands": 6}))
    logs = {}
    for mode in ("normal", "muted", "shuffled", "false-claims"):
        players = [HouseStylePlayer(f"seat{i}", style, 13, talk_mode=mode) for i, style in enumerate(HOUSE_STYLES)]
        logs[mode] = await play_session(mode, 13, players, rules)
    actions = {mode: [a for hand in log["hands"] for a in hand["actions"]] for mode, log in logs.items()}
    baseline = [(a["player"], a["action"]) for a in actions["normal"]]
    assert all([(a["player"], a["action"]) for a in moves] == baseline for moves in actions.values())
    assert any(a["say"] for a in actions["normal"])
    assert all(not a["say"] for a in actions["muted"])
    assert [a["say"] for a in actions["shuffled"]] != [a["say"] for a in actions["normal"]]
    claims = [
        (action, hand["starting_state"][action["player"]]["hole_cards"])
        for hand in logs["false-claims"]["hands"] for action in hand["actions"] if action["say"]
    ]
    assert claims
    for action, cards in claims:
        assert action["street"] == "preflop" and action["action"]["type"] == "raise"
        claimed_rank = "A" if "aces" in action["say"] else "K"
        assert any(card[0] != claimed_rank for card in cards)


@pytest.mark.parametrize(("arm", "talk_mode"), [
    ("opponent_reads", "shuffled"),
    ("table_talk_defense", "shuffled"),
    ("table_talk_defense", "false-claims"),
    ("balanced_play", "shuffled"),
    ("candidate", "shuffled"),
])
def test_ablation_plays_paired_games_and_keeps_logs_separate(monkeypatch, tmp_path, capsys, arm, talk_mode) -> None:
    app = FastAPI()
    systems = set()
    models = set()

    @app.post("/v1/chat/completions")
    async def complete(request: Request) -> dict:
        body = await request.json()
        systems.add(body["messages"][0]["content"].partition("\n\nFor this blind defense")[0])
        models.add(body["model"])
        cleanup = "Now call update_scratchpad" in body["messages"][-1].get("content", "")
        name = "update_scratchpad" if cleanup else "submit_action"
        args = {"contents": "seat 3 uncertain"} if cleanup else {"action": "call"}
        call = {"id": "call_1", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
        return {"choices": [{"finish_reason": "tool_calls", "message": {"tool_calls": [call]}}], "usage": {}}

    client_class = httpx.AsyncClient

    def local_client(**kwargs):
        return client_class(transport=httpx.ASGITransport(app=app), **kwargs)

    monkeypatch.setattr(score_local.httpx, "AsyncClient", local_client)
    monkeypatch.setattr(score_local, "GATEWAY_URL", "http://gw/v1")
    monkeypatch.setenv("AC2_API_KEY", "test-key")
    flags = {
        "opponent_reads": ["--opponent-read-ablation"],
        "table_talk_defense": ["--table-talk-defense-ablation"],
        "balanced_play": ["--balanced-play-ablation"],
        "candidate": ["--compare-model", "p/base"],
    }
    lineup = "arena" if arm == "candidate" else "mixed"
    monkeypatch.setattr(
        sys, "argv",
        ["score_local.py", "--model", "p/entry", "--games", "1", "--hands", "3", "--seed", "13", "--lineup", lineup,
         "--talk-mode", talk_mode, "--out", str(tmp_path), *flags[arm]],
    )
    score_local.main()

    control = json.loads((tmp_path / "control" / "score-13.json").read_text())
    treatment = json.loads((tmp_path / arm / "score-13.json").read_text())
    assert control["lineup"] == treatment["lineup"] and control["scored_seat"] == treatment["scored_seat"]
    if talk_mode == "false-claims":
        assert len(systems) == 3 and (tmp_path / "specific_claims" / "score-13.json").is_file()
        assert sum(SPECIFIC_CLAIMS in system for system in systems) == 1
    else:
        assert len(systems) == (1 if arm == "candidate" else 2)
    assert models == ({"p/entry", "p/base", "vals-hackathon/entry"} if arm == "candidate" else {"p/entry"})
    if lineup == "arena":
        assert list(control["lineup"].values()).count("field") == 4
    if arm == "opponent_reads":
        assert all(PLACEMENT_GUIDANCE in system for system in systems)
        assert sum(ACTION_GUIDANCE in system and CLEANUP_GUIDANCE in system for system in systems) == 1
    elif arm == "table_talk_defense":
        assert sum(ACTION_DEFENSE in system and CLEANUP_DEFENSE in system for system in systems) == 1
    elif arm == "balanced_play":
        assert sum(BALANCED_PLAY in system for system in systems) == 1
    assert score_local.summarize(treatment)["cleanup_updated"] >= 1
    paired = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert paired["paired_games"] == 1 and paired["mean_places_gained"] == 0 and paired["tied"] == 1
    assert paired["rating_points_gained_per_game"] == 0
    if talk_mode == "false-claims":
        assert paired["comparison"] == "ignore_vs_specific_claims"
