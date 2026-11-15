from subprocess import CompletedProcess

import pytest

import checkpoint_loop
import eval_suite


def test_failed_checkpoint_deploy_only_cleans_up_its_own_id(monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if "--checkpoint" in command:
            return CompletedProcess(command, 1, "deploying poker/tmp-s5-1234 as infer_fixture\n", "probe failed")
        assert kwargs["check"] is True
        return CompletedProcess(command, 0)

    monkeypatch.setattr(checkpoint_loop.subprocess, "run", run)
    assert checkpoint_loop.deploy("poker", "tmp-s5-1234", "ckpt_fixture") is None
    assert calls[1][-6:] == ["--project", "poker", "--name", "tmp-s5-1234", "--delete-id", "infer_fixture"]


def test_checkpoint_cannot_win_the_screen_with_a_broken_arm():
    notes = {"vs_baseline": 100, "decisions": 100, "invalid": 0}
    candidate = {"calls_notes": notes, "calls_empty": {"decisions": 100, "invalid": 6}}
    assert not checkpoint_loop.eligible(candidate)
    candidate["calls_empty"]["invalid"] = 5
    assert checkpoint_loop.eligible(candidate)


def test_call_interval_keeps_decisions_from_each_game_together():
    values = {f"game-a:1:0:{i}": 0.3 for i in range(100)} | {"game-b:1:0:0": -0.4}
    mean, low, high = eval_suite.bootstrap_calls(values)
    assert mean == pytest.approx(1000 * (30 - 0.4) / 101)
    assert (low, high) == pytest.approx((-400, 300))
