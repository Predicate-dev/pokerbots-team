import json

import score_local
from poker_harness.arena_lineups import recent_rosters, replay_roster
from poker_harness.opponent_reads import betting_evidence
from poker_harness.seat import Completion


def match(match_id="m_first", finished="2026-11-15T20:00:00Z"):
    return {
        "id": match_id,
        "kind": "ranked",
        "finished_at": finished,
        "seats": [
            {"seat": 0, "entry": {"kind": "team", "id": "org_maple"}},
            {"seat": 1, "entry": {"kind": "house", "id": "house-anxious-alice"}},
            {"seat": 2, "entry": {"kind": "team", "id": "org_friends", "name": "Friends"}},
            {"seat": 3, "entry": {"kind": "house", "id": "house:calling-station"}},
            {"seat": 4, "entry": {"kind": "team", "id": "org_cork"}},
            {"seat": 5, "entry": {"kind": "house", "id": "house-bob-the-buster"}},
        ],
    }


def test_recent_rosters_and_training_keep_the_recorded_house_seats(tmp_path):
    for filename, game in (
        ("1.json", match()),
        ("2.json", match("m_second", "2026-11-15T20:05:00Z")),
        ("3.json", match("m_exhibition", "2026-11-15T20:08:00Z") | {"kind": "exhibition"}),
    ):
        (tmp_path / filename).write_text(json.dumps(game))
    rosters = recent_rosters(tmp_path, "org_friends", limit=1)
    assert rosters[0].match_id == "m_second"
    assert rosters[0].hero == 2
    assert rosters[0].opponent_specs([200, 30, 370, 0, 100, 500]) == {
        "0": {"thinking": False},
        "1": "scripted:house_alice",
        "4": {"thinking": False},
        "5": "scripted:house_bob",
    }


async def test_deployment_score_uses_recorded_house_roster_and_keeps_official_seat_names(monkeypatch):
    async def reply(messages):
        if "Legal actions: fold, call" in messages[-1]["content"]:
            return Completion(text="", tool_arguments={"action": "fold"})
        return Completion(text="", tool_arguments={"action": "check"})

    monkeypatch.setattr(score_local, "relay_completion", lambda *args: reply)
    roster = replay_roster(match(), "Friends")
    a = await score_local.one_game(
        None,
        "fixture",
        "test/a",
        77,
        1,
        lineup="arena",
        field_model="test/field",
        arena_rosters=(roster,),
    )
    b = await score_local.one_game(
        None,
        "fixture",
        "test/b",
        77,
        1,
        lineup="arena",
        field_model="test/field",
        arena_rosters=(roster,),
    )
    assert a["scored_seat"] == "seat2"
    assert a["house_opponents"] == {"seat1": "house_alice", "seat3": "calling_station", "seat5": "house_bob"}
    assert set(a["hands"][0]["starting_state"]) == {f"seat{i}" for i in range(6)}
    assert score_local.summarize(a)["house_count"] == 3
    assert a["matchup"] == b["matchup"]


def test_house_evidence_counts_actions_once_per_hand_and_ignores_chatter():
    turn = (
        "Your seat: 2\n"
        '  preflop: seat1 (seat 1) raise to 120 and says: "oh no, sorry, i fold"\n'
        '  preflop: seat3 (seat 3) call and says: "all in, baby"\n'
        "  river: seat3 (seat 3) call\n"
        "  preflop: seat2 (seat 2) fold\n"
    )
    first = betting_evidence([turn, turn])
    assert first.splitlines() == [
        "seat 1 | preflop 1/1 played | opens/raises 1 | big raises 1 | postflop calls 0 | postflop bets/raises 0",
        "seat 3 | preflop 1/1 played | opens/raises 0 | big raises 0 | postflop calls 1 | postflop bets/raises 0",
    ]
    second = betting_evidence(["Your seat: 2\n  preflop: seat1 (seat 1) fold"], first)
    assert "seat 1 | preflop 1/2 played" in second
    assert "seat 3 | preflop 1/1 played" in second
