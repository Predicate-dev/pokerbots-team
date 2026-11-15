"""The training-only task builders (`vals_tools/training_tasks.py`, `rl/make_training_datasets.py`), checked against
this repo's harness: every task they build passes the harness's own validation, and one of each multi-hand family is
played and graded end to end."""

import asyncio
import importlib.util
import random
import sys
from pathlib import Path

import httpx
import pytest

from vals_tools.tasks import SPLITS, TABLE_RULES, Opponents, hand_tasks
from vals_tools.training_tasks import (
    GAME_HANDS,
    HARNESS_SRC,
    MIN_LIVE_STACK,
    PERSONAS,
    endgame_stacks,
    endgame_tasks,
    mixed_table_tasks,
    strong_hand_tasks,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HARNESS_SRC), str(ROOT / "harness" / "tests")]

from poker_harness.completions import table_rules, with_preset  # noqa: E402
from poker_harness.grading import check_grader_params  # noqa: E402
from poker_harness.server import create_app  # noqa: E402
from poker_harness.spots import chen_score, hole_cards  # noqa: E402
from poker_harness.training import opponent_specs, table_start  # noqa: E402
from test_grading import grade, grading_body  # noqa: E402
from test_server import play  # noqa: E402
from test_training import event_relay  # noqa: E402

OPPONENT = Opponents("my-project/opponent", "chat", None)


def load_builder():
    spec = importlib.util.spec_from_file_location("make_training_datasets", ROOT / "rl" / "make_training_datasets.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_strong_hand_tasks_keep_playable_hands_and_a_share_of_weak_ones() -> None:
    tasks = strong_hand_tasks(240, 7_710_001, OPPONENT, min_chen=6, weak_fraction=0.15)
    assert [t.env_params for t in tasks] == [t.env_params for t in strong_hand_tasks(240, 7_710_001, OPPONENT)]
    assert [t.env_params["policy_seats"][0] for t in tasks[:12]] == [0, 1, 2, 3, 4, 5] * 2
    scores = [chen_score(hole_cards(t.env_params["seed"], t.env_params["policy_seats"][0], seats=6)) for t in tasks]
    weak = sum(score < 6 for score in scores)
    assert 0.07 * 240 <= weak <= 0.25 * 240  # about the 15% asked for
    assert strong_hand_tasks(60, 1, OPPONENT, weak_fraction=0.0) and all(
        chen_score(hole_cards(t.env_params["seed"], t.env_params["policy_seats"][0], seats=6)) >= 8
        for t in strong_hand_tasks(60, 1, OPPONENT, min_chen=8, weak_fraction=0.0)
    )
    # The same task shape as the official single-hand task, and no deal the official splits use.
    official = hand_tasks(1, 5, OPPONENT)[0].env_params
    assert set(tasks[0].env_params) == set(official) and tasks[0].grader_params is None
    seeds = {t.env_params["seed"] for t in tasks}
    used = {t.env_params["seed"] for n, base in SPLITS.values() for t in hand_tasks(n, base, OPPONENT)}
    assert len(seeds) == len(tasks) and not seeds & used
    with pytest.raises(ValueError, match="weak_fraction"):
        strong_hand_tasks(1, 1, OPPONENT, weak_fraction=1.0)


def test_endgame_stacks_keep_every_chip_and_cover_short_and_big_stacks() -> None:
    rng = random.Random(3)
    seen = []
    for i in range(600):
        seat = i % 6
        stacks = endgame_stacks(rng, seat)
        live = [chips for chips in stacks if chips]
        assert len(stacks) == 6 and sum(stacks) == 1200 and stacks[seat] > 0
        assert len(live) >= 2 and min(live) >= MIN_LIVE_STACK
        seen.append((stacks[seat], 6 - len(live)))
    mine, busted = zip(*seen, strict=True)
    assert min(mine) <= 60 and max(mine) >= 360 and {0, 1, 2} <= set(busted)


def test_every_training_task_passes_the_harness_validation() -> None:
    endgames = endgame_tasks(120, 7_730_003, OPPONENT, max_hands_left=4, chip_weight=0.3)
    mixed = mixed_table_tasks(120, 7_750_005, OPPONENT, hands=3)
    for task in [*endgames, *mixed]:
        params = with_preset(task.env_params)
        rules = table_rules(params)
        policy = set(params["policy_seats"])
        start = table_start(params, rules, policy)
        specs = opponent_specs(params, rules.seats, policy)
        check_grader_params(task.grader_params)
        assert rules.total_hands == GAME_HANDS and not rules.reset_stacks and rules.scratchpad
        assert policy.isdisjoint(specs) and task.env_params["preset"] == TABLE_RULES["preset"]
        hands = (start.last_hand or GAME_HANDS) - start.first_hand + 1
        assert 1 <= hands <= 4

    assert {t.env_params["first_hand"] for t in endgames} == {27, 28, 29, 30}
    assert all(t.grader_params == {"reward": "placement", "chip_weight": 0.3} for t in endgames)
    assert endgame_tasks(1, 1, OPPONENT)[0].grader_params == {"reward": "placement"}
    assert all(t.env_params["last_hand"] == 3 and t.grader_params == {"reward": "ev", "ev_unit_bb": 3} for t in mixed)
    kinds = [spec for t in mixed for spec in t.env_params["opponent_seats"].values()]
    assert any(isinstance(k, str) for k in kinds) and any(isinstance(k, dict) for k in kinds)
    assert {k["persona"] for k in kinds if isinstance(k, dict)} <= set(PERSONAS.values())
    assert any(len(t.env_params["opponent_seats"]) < 5 for t in mixed)  # some seats stay the plain opponent model
    with pytest.raises(ValueError, match="hands"):
        mixed_table_tasks(1, 1, OPPONENT, hands=5)
    with pytest.raises(ValueError, match="max_hands_left"):
        endgame_tasks(1, 1, OPPONENT, max_hands_left=5)


async def _play_and_grade(task, tmp_path: Path) -> tuple[dict, dict, dict, list[dict]]:
    policy_requests: list[dict] = []
    opponent_requests: list[dict] = []
    app = create_app(
        relay_transport=httpx.ASGITransport(app=event_relay(policy_requests)),
        opponent_transport=httpx.ASGITransport(app=event_relay(opponent_requests)),
        runs_dir=tmp_path,
    )
    env_params = task.env_params
    status, outputs = await play(app, {"input": [], "env_params": env_params})
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://h") as client:
        body = grading_body("gw_1", env_params=env_params, grader_params=task.grader_params)
        graded, score = await grade(client, body)
    assert graded["status"] == "completed", graded
    return status, outputs, score.json(), opponent_requests


def test_an_endgame_task_plays_and_is_graded_on_place(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AC2_API_KEY", "gateway-key")
    monkeypatch.setenv("POKER_OPPONENT_BASE_URL", "http://relay/job/trace/v1")
    task = next(t for t in endgame_tasks(30, 7_730_003, OPPONENT) if t.env_params["first_hand"] == 28)
    status, outputs, score, opponent_requests = asyncio.run(_play_and_grade(task, tmp_path))
    log = outputs["log"]
    assert status["status"] == "completed", status
    assert log["hands"][0]["hand_number"] == 28 and log["hands"][-1]["hand_number"] <= 30
    assert log["session_info"]["starting_stacks"] == task.env_params["initial_stacks"]
    assert sum(log["final_stacks"].values()) == 1200 and opponent_requests
    assert 1.0 <= score["score"] <= 11.0 and score["artifacts"]["reward"] == "placement"
    assert 1 <= score["artifacts"]["place"] <= 6


def test_a_mixed_table_task_plays_and_is_graded_on_ev(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AC2_API_KEY", "gateway-key")
    monkeypatch.setenv("POKER_OPPONENT_BASE_URL", "http://relay/job/trace/v1")
    tasks = mixed_table_tasks(40, 7_750_005, OPPONENT, hands=2)
    task = next(  # scripted and persona seats at one table; no maniac, whose all-ins can end the game on hand 1
        t
        for t in tasks
        if any(isinstance(s, str) for s in t.env_params["opponent_seats"].values())
        and any(isinstance(s, dict) for s in t.env_params["opponent_seats"].values())
        and "scripted:maniac" not in t.env_params["opponent_seats"].values()
    )
    status, outputs, score, opponent_requests = asyncio.run(_play_and_grade(task, tmp_path))
    log = outputs["log"]
    assert status["status"] == "completed", status
    assert [h["hand_number"] for h in log["hands"]] == [1, 2] and log["session_info"]["game_hands"] == 30
    scripted_seats = {f"seat{s}" for s, spec in task.env_params["opponent_seats"].items() if isinstance(spec, str)}
    providers = {a["player"]: a["model_response"]["provider"] for h in log["hands"] for a in h["actions"]}
    assert all(providers[name] == "scripted" for name in scripted_seats if name in providers)
    personas = {s["persona"] for s in task.env_params["opponent_seats"].values() if isinstance(s, dict)}
    systems = {r["messages"][0]["content"] for r in opponent_requests}
    assert all(any(system.endswith(persona) for system in systems) for persona in personas)
    assert 1.0 <= score["score"] <= 11.0 and score["artifacts"]["hands"] == 2 and "ev_bb" in score["artifacts"]


def test_the_builder_prints_by_default_and_names_train_and_select_per_family(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    builder = load_builder()
    args = builder.parser().parse_args(["--count", "12", "--select-count", "6"])
    built = builder.build(args)
    assert list(built) == [
        f"pokerbots-{family}-{split}" for family in ("strong", "endgame", "mixed", "house") for split in ("train", "select")
    ]
    assert [len(tasks) for tasks in built.values()] == [12, 6] * 4
    train, select = built["pokerbots-endgame-train"], built["pokerbots-endgame-select"]
    assert not {t.env_params["seed"] for t in train} & {t.env_params["seed"] for t in select}

    def no_client(*args: object, **kwargs: object) -> None:
        raise AssertionError("a dry run must not reach AC2")

    monkeypatch.setattr("ac2.sdk.Client", no_client)
    monkeypatch.setattr(sys, "argv", ["make_training_datasets.py", "--family", "endgame", "--count", "3", "--select-count", "2"])
    builder.main()
    out = capsys.readouterr().out
    assert "pokerbots-endgame-train 3" in out and "pokerbots-endgame-select 2" in out
    assert "nothing uploaded" in out and "pokerbots-strong" not in out
    monkeypatch.setattr(sys, "argv", ["make_training_datasets.py", "--upload", "--dry-run"])
    with pytest.raises(SystemExit):
        builder.main()


def test_the_select_split_defaults_to_a_quarter_of_the_train_count_and_zero_means_zero() -> None:
    builder = load_builder()
    built = builder.build(builder.parser().parse_args(["--family", "endgame", "--count", "8"]))
    assert [len(tasks) for tasks in built.values()] == [8, 2]
    built = builder.build(builder.parser().parse_args(["--family", "endgame", "--count", "4", "--select-count", "0"]))
    assert [len(tasks) for tasks in built.values()] == [4, 0]


def test_house_tables_are_all_scripted_windows_of_a_30_hand_game() -> None:
    from vals_tools.training_tasks import HOUSE_STYLES, house_table_tasks

    tasks = house_table_tasks(60, 5, OPPONENT)
    assert [t.env_params["policy_seats"] for t in tasks[:6]] == [[i] for i in range(6)]
    fresh = 0
    for task in tasks:
        params = task.env_params
        seat = params["policy_seats"][0]
        lineup = params["opponent_seats"]
        assert sorted(lineup) == [str(s) for s in range(6) if s != seat]
        assert all(spec.removeprefix("scripted:") in HOUSE_STYLES for spec in lineup.values())
        assert len(set(lineup.values())) == 5
        assert params["total_hands"] == 30 and params["first_hand"] == params["last_hand"]
        if "initial_stacks" in params:
            assert sum(params["initial_stacks"]) == 1200 and params["initial_stacks"][seat] > 0
        else:
            fresh += params["first_hand"] == 1
        assert task.grader_params == {"reward": "ev", "ev_unit_bb": 1}
    assert 5 <= fresh <= 35
    placed = house_table_tasks(6, 5, OPPONENT, hands=3, placement=True)
    assert all(t.grader_params == {"reward": "placement"} and "ev_labels" not in t.env_params for t in placed)
