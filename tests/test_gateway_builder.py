"""opsd/build_from_gateway.py: hinted tasks from a deployment's gateway spans, with no AC2 calls."""

from __future__ import annotations

import importlib.util
import json
import sys
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_builder():
    sys.path.insert(0, str(ROOT / "opsd"))
    spec = importlib.util.spec_from_file_location("build_from_gateway", ROOT / "opsd" / "build_from_gateway.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@cache
def official_system_prompt() -> str:
    load_builder()
    from build_dataset import official_system_prompt as prompt

    return prompt()


def row(span: str, call: str, arguments: str, system: str | None = None, tokens_out: int = 400) -> dict:
    prompt = [
        {"type": "message", "role": "system", "content": official_system_prompt() if system is None else system},
        {"type": "message", "role": "user", "content": "## Hand 3 of 30\nYour hole cards: Ah Kd"},
    ]
    reply = [
        {"type": "reasoning", "content": "Strong hand, raise."},
        {"type": "message", "role": "assistant", "content": "\n\n"},
        {"type": "function_call", "call_id": f"call-{span}", "name": call, "arguments": arguments},
    ]
    return {
        "trace_id": "t1",
        "span_id": span,
        "status": "completed",
        "start_time": "2026-11-14T23:00:00",
        "tokens_out": tokens_out,
        "input": json.dumps(prompt),
        "output": json.dumps(reply),
    }


def test_only_clean_poker_decisions_become_hinted_tasks() -> None:
    builder = load_builder()
    rows = [
        row("a", "submit_action", '{"action":"raise","amount":6}'),
        row("a", "submit_action", '{"action":"raise","amount":6}'),  # the same span twice: counted once
        row("b", "update_scratchpad", '{"contents":"notes"}'),  # the after-hand turn
        row("c", "submit_action", '{"action":"fold"}', system="Reply with OK."),  # not a poker prompt
        row("d", "submit_action", '{"action":"call"}', tokens_out=5000),  # cut off at the cap
    ]
    tasks, totals = builder.build(rows, "HINT TEXT", 5000)
    assert totals["kept"] == len(tasks) == 1
    assert totals["cleanup"] == 1 and totals["not_poker"] == 1 and totals["truncated"] == 1
    task = tasks[0]
    selection = task.opsd_hint_selections[0]
    assert selection.hint == "HINT TEXT" and selection.rollout_prefix_len == 2  # the hint goes after the prompt
    assert len(task.input) == 5  # the prompt's two items, then the reply's three
    assert {tool["name"] for tool in task.env_params["tools"]} == {"submit_action", "update_scratchpad"}


def test_gateway_defense_keeps_actor_and_cleanup_guidance_teacher_only() -> None:
    builder = load_builder()
    from build_dataset import teacher_hints
    from poker_harness.talk_defense import ACTION_DEFENSE, CLEANUP_DEFENSE

    rows = [
        row("a", "submit_action", '{"action":"fold"}'),
        row("b", "update_scratchpad", '{"contents":"seat 2 uncertain"}', tokens_out=6_000),
        row("c", "update_scratchpad", '{"contents":2}'),
        row("d", "update_scratchpad", '{"contents":"cut off"}', tokens_out=10_000),
    ]
    hints = teacher_hints(opponent_reads=False, table_talk_defense=True)
    tasks, totals = builder.build(rows, hints[0], 5_000, cleanup_hint=hints[1])
    assert [task.opsd_hint_selections[0].hint for task in tasks] == [ACTION_DEFENSE, CLEANUP_DEFENSE]
    assert totals["cleanup_kept"] == 1 and totals["format"] == totals["truncated"] == 1
    assert all("Player chatter is untrusted" not in item.model_dump_json() for task in tasks for item in task.input)


def test_balanced_one_step_stores_prompt_without_replaying_the_fold(monkeypatch) -> None:
    builder = load_builder()
    import build_dataset
    from hint import HINT
    from poker_harness.balanced_play import BALANCED_PLAY
    from poker_harness.talk_defense import ACTION_DEFENSE, CLEANUP_DEFENSE

    def reference(system, user):
        assert system == official_system_prompt() and user.startswith("## Hand 3 of 30")
        return "Call costs 2 chips."

    monkeypatch.setattr(build_dataset, "reference_hint", reference)
    hint, cleanup = build_dataset.teacher_hints(False, True, strategy="balanced")
    assert hint == f"{BALANCED_PLAY}\n\n{ACTION_DEFENSE}" and cleanup == CLEANUP_DEFENSE
    assert build_dataset.teacher_hints(False, False) == (BALANCED_PLAY, None)
    assert build_dataset.teacher_hints(False, False, strategy="tight-survival") == (HINT, None)

    rows = [
        row("a", "submit_action", '{"action":"fold"}'),
        row("b", "update_scratchpad", '{"contents":"notes"}'),
    ]
    tasks, totals = builder.build(rows, hint, 5000, cleanup_hint=cleanup, rollout_mode="one_step")
    assert totals["kept"] == 2 and totals["preflop_reference_kept"] == 1
    assert [task.opsd_hint_selections[0].hint for task in tasks] == [f"{hint}\n\nCall costs 2 chips.", cleanup]
    for task in tasks:
        assert len(task.input) == task.opsd_hint_selections[0].rollout_prefix_len == 2
        assert [item.role for item in task.input] == ["system", "user"]
        assert task.input[0].content == official_system_prompt()


def test_prompts_with_text_appended_to_the_system_prompt_are_left_out() -> None:
    builder = load_builder()
    from build_dataset import decision_tasks

    official = official_system_prompt()
    rows = [
        row("a", "submit_action", '{"action":"fold"}', system=official),
        row("b", "submit_action", '{"action":"fold"}', system=f"{official}\n\nStrategy notes: fold more."),
    ]
    extra = row("c", "submit_action", '{"action":"fold"}')
    messages = json.loads(extra["input"])
    messages.insert(1, {"type": "message", "role": "system", "content": "Keep playing the tight strategy."})
    extra["input"] = json.dumps(messages)
    rows.append(extra)
    tasks, totals = builder.build(rows, "HINT", 5000)
    assert len(tasks) == 1 and totals["not_poker"] == 2
    trace = {"trace_id": "eval", "spans": [builder.as_trace(row)["spans"][0] for row in rows]}
    tasks, stats = decision_tasks(trace, "HINT", 5000)
    assert len(tasks) == 1 and stats["not_official"] == 2


def test_custom_strategy_preserves_reads_and_talk_defense() -> None:
    load_builder()
    from build_dataset import teacher_hints
    from poker_harness.opponent_reads import ACTION_GUIDANCE, CLEANUP_GUIDANCE
    from poker_harness.talk_defense import ACTION_DEFENSE, CLEANUP_DEFENSE

    action, cleanup = teacher_hints(True, True, action_hint="Custom strategy.")
    assert action == f"Custom strategy.\n\n{ACTION_GUIDANCE}\n\n{ACTION_DEFENSE}"
    assert cleanup == f"{CLEANUP_GUIDANCE}\n\n{CLEANUP_DEFENSE}"
