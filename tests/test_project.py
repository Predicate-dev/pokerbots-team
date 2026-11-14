import importlib.util
import json
import sys
from pathlib import Path

import pytest
from ac2.runtime import FunctionCall, Message, Reasoning

from vals_tools.score import summarize
from vals_tools.tasks import GAME_HANDS, SPLITS, Opponents, datasets, placement_datasets, snapshot_datasets

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "opsd")]

import build_dataset  # noqa: E402
import train as opsd_train  # noqa: E402

OPPONENT = Opponents("my-project/opponent")


def test_splits_share_no_deal_and_cycle_every_seat() -> None:
    built = datasets("p", OPPONENT)
    seeds = [t.env_params["seed"] for name, tasks in built.items() if not name.endswith("table-smoke") for t in tasks]
    assert len(seeds) == len(set(seeds)) == sum(n for n, _ in SPLITS.values())
    train = built["p-train"]
    assert {t.env_params["policy_seats"][0] for t in train[:6]} == set(range(6))
    params = train[0].env_params
    assert params["seats"] == 6 and params["opponent_model"] == "my-project/opponent" and params["ev_labels"]
    assert params["opponent_api"] == "responses" and params["opponent_reasoning_effort"] == "medium"
    assert params["preset"] == "event" and built["p-table-smoke"][0].env_params["preset"] == "event"
    assert params["total_hands"] == 1 and "chat_template_kwargs" not in params  # thinking comes from the preset
    assert "policy_seats" not in built["p-table-smoke"][0].env_params  # every seat is the evaluated model


def test_placement_datasets_mix_mid_game_hands_scored_on_place_with_fresh_hands_scored_on_chips() -> None:
    built = placement_datasets("p-place", OPPONENT)
    assert set(built) == {f"p-place-{split}" for split in SPLITS}
    train = built["p-place-train"]
    assert placement_datasets("p-place", OPPONENT)["p-place-train"][7].env_params == train[7].env_params  # same seeds
    midgame = [t for t in train if t.grader_params]
    fresh = [t for t in train if not t.grader_params]
    assert 0.4 < len(midgame) / len(train) < 0.6
    assert all(t.env_params["total_hands"] == 1 and "starting_stacks" not in t.env_params for t in fresh)
    late = 0
    for task in midgame:
        params, hand = task.env_params, task.env_params["first_hand"]
        stacks, seat = params["starting_stacks"], params["policy_seats"][0]
        assert 2 <= hand <= GAME_HANDS and params["play_hands"] == 1 and "total_hands" not in params  # the preset's 30
        assert task.grader_params == {"reward": "placement", "hands_left": GAME_HANDS - hand}
        assert len(stacks) == 6 and sum(stacks) == 1200 and min(stacks) >= 0
        assert stacks[seat] > 0 and sum(s > 0 for s in stacks) >= 2
        late += hand >= 24
    assert 0.4 < late / len(midgame) < 0.6 and any(0 in t.env_params["starting_stacks"] for t in midgame)
    seeds = [t.env_params["seed"] for tasks in built.values() for t in tasks]
    assert len(seeds) == len(set(seeds))


def test_ledger_datasets_seed_a_ledger_that_matches_the_table_and_grade_its_update() -> None:
    train = placement_datasets("p-ledger", OPPONENT, ledger=True)["p-ledger-train"]
    assert all(t.grader_params["ledger_weight"] == 2 for t in train)
    midgame = [t for t in train if "first_hand" in t.env_params]
    fresh = [t for t in train if "first_hand" not in t.env_params]
    for task in midgame:
        params, ledger = task.env_params, task.env_params["starting_ledger"]
        assert task.grader_params["reward"] == "placement"
        assert (ledger["hand"], ledger["total"]) == (params["first_hand"] - 1, GAME_HANDS)
        assert ledger["me"] == f"seat{params['policy_seats'][0]}"
        assert [row["stack"] for row in ledger["seats"].values()] == params["starting_stacks"]
        live = [row for row in ledger["seats"].values() if row["stack"]]
        assert all(row["n"] == ledger["hand"] and row["3b"] <= row["pr"] <= row["vp"] <= row["n"] for row in live)
    seeded = [t for t in fresh if "starting_ledger" in t.env_params]
    assert 0.4 < len(seeded) / len(fresh) < 0.6  # the rest start blank, as every arena game does
    assert all(t.env_params["starting_ledger"]["hand"] == 0 and "reward" not in t.grader_params for t in seeded)
    plain = placement_datasets("p-place", OPPONENT)["p-place-train"]
    assert not any("starting_ledger" in t.env_params or "ledger_weight" in (t.grader_params or {}) for t in plain)


def test_snapshot_datasets_replay_real_hands_and_keep_each_game_in_one_split() -> None:
    def spot(game: int, hand: int) -> dict:
        ledger = {"hand": hand - 1, "total": 30, "me": "seat2", "seats": {"seat0": {"stack": 300}}}
        return {
            "game_id": f"game{game}",
            "seed": 1000 + game,
            "seats": 6,
            "first_hand": hand,
            "play_hands": 1,
            "button_seat": hand % 6,
            "starting_stacks": [300, 100, 200, 200, 0, 400],
            "policy_seats": [2],
            "hands_left": 30 - hand,
        } | ({"starting_ledger": ledger} if hand > 1 else {})

    spots = [spot(game, hand) for game in range(40) for hand in (1, 12, 30)]
    built = snapshot_datasets("p-real", spots, OPPONENT)
    train, select = built["p-real-train"], built["p-real-select"]
    assert len(train) + len(select) == 120 and 0 < len(select) < len(train)
    assert not {t.env_params["seed"] for t in train} & {t.env_params["seed"] for t in select}  # no game in both
    assert [t.env_params["seed"] for t in train] != sorted(t.env_params["seed"] for t in train)  # games are mixed
    for task in train + select:
        params, hand = task.env_params, task.env_params["first_hand"]
        assert params["preset"] == "event" and params["opponent_model"] == "my-project/opponent"
        assert params["button_seat"] == hand % 6 and params["starting_stacks"] == [300, 100, 200, 200, 0, 400]
        assert task.grader_params == {"reward": "placement", "hands_left": 30 - hand, "ledger_weight": 2}
        assert ("starting_ledger" in params) == (hand > 1) and "game_id" not in params and "hands_left" not in params
    plain = snapshot_datasets("p-real", spots, OPPONENT, ledger=False)["p-real-train"]
    assert all("starting_ledger" not in t.env_params and "ledger_weight" not in t.grader_params for t in plain)


def row(ev: float) -> dict:
    return {"ev_bb": ev, "realized_bb": ev, "policy_format_errors": 0}


def test_summary_pairs_by_deal() -> None:
    rows = {"a": row(-0.5), "b": row(4.0), "c": row(2.0)}
    base = {"a": row(-3.0), "b": row(1.0), "c": row(1.0)}
    out = summarize(rows, base)
    assert out["hands"] == 3 and out["ev_bb_per_hand"] == pytest.approx(5.5 / 3, abs=1e-3)
    assert out["vs_base"]["paired_hands"] == 3 and out["vs_base"]["diff"] == pytest.approx(6.5 / 3, abs=1e-3)


def test_opsd_builder_makes_one_hinted_task_per_clean_decision() -> None:
    system = Message(role="system", content="rules")
    user = Message(role="user", content="state")
    good = [Reasoning(content="think"), FunctionCall(call_id="c1", name="submit_action", arguments='{"action":"fold"}')]
    bad = [Message(role="assistant", content="no tool call")]

    def span(start: int, reply: list) -> dict:
        dump = lambda items: json.dumps([i.model_dump(mode="json") for i in items])  # noqa: E731
        return {
            "name": "get_completion",
            "status": "OK",
            "start_time": start,
            "input": dump([system, user]),
            "output": dump(reply),
            "tokens_out": 10,
        }

    tasks, stats = build_dataset.decision_tasks({"trace_id": "t", "spans": [span(1, good), span(2, bad)]}, "hint", 100)
    assert stats == {"decisions": 2, "kept": 1, "cleanup": 0, "format": 1, "truncated": 0}
    (task,) = tasks
    (selection,) = task.opsd_hint_selections
    assert len(task.input) == 4 and selection.rollout_prefix_len == 2
    assert selection.mutation.item_boundary == 1 and selection.mutation.action == "append_to_previous_item"
    assert [t["name"] for t in task.env_params["tools"]] == ["submit_action", "update_scratchpad"]
    assert set(task.env_params["tools"][0]["parameters"]["properties"]) == {"action", "amount", "say"}
    assert "hint" not in json.dumps([i.model_dump(mode="json") for i in task.input])  # only the teacher sees it


def test_training_configs_send_only_explicit_settings() -> None:
    """Unset flags are not sent, so AC2 keeps its defaults; a flag that is set shows up in the configuration."""
    spec = importlib.util.spec_from_file_location("rl_train", ROOT / "rl" / "train.py")
    grpo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(grpo)
    config = grpo.run_config(grpo.parser().parse_args([]), "n").to_json()
    text = json.dumps(config, default=str)
    assert "dr-grpo" in text and "pokerbots-train" in text and "pokerbots-select" in text
    assert "learning_rate" not in text
    # The hand's token budget is set explicitly: AC2's default (8,192) runs out on multi-decision hands.
    assert config["experiment_cfg"]["rollout"]["max_response_len"] == 24576
    # Replica counts are set: AC2's default for this model (4 + 4 replicas) needs more GPUs than a team's quota.
    assert config["experiment_cfg"]["rollout"]["n_inference_replicas"] == 1
    assert config["experiment_cfg"]["eval"]["n_inference_replicas"] == 1
    tuned = grpo.run_config(grpo.parser().parse_args(["--lr", "1e-6", "--max-response-len", "10000"]), "n").to_json()
    assert tuned["experiment_cfg"]["trainer"]["learning_rate"] == 1e-6
    assert tuned["experiment_cfg"]["rollout"]["max_response_len"] == 10000

    opsd = opsd_train.run_config(opsd_train.parser().parse_args(["--dataset", "d"]), "n").to_json()
    text = json.dumps(opsd, default=str)
    assert "offline" in text and '"num_train_steps": 20' in text
    assert "learning_rate" not in text and "max_response_len" not in text
    assert opsd["experiment_cfg"]["rollout"] == {"n_inference_replicas": 0}


def test_every_deployment_is_exactly_one_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """One replica, never scaled out, and no flag to ask for more."""
    from vals_tools import deploy

    assert (deploy.SCALING.min_replicas, deploy.SCALING.max_replicas) == (1, 1)
    monkeypatch.setattr(sys, "argv", ["deploy.py", "--name", "x", "--model", "m", "--replicas", "2"])
    with pytest.raises(SystemExit) as exit_info:
        deploy.main()
    assert exit_info.value.code == 2  # argparse rejects the unknown flag before anything reaches AC2
