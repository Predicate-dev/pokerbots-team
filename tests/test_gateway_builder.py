"""opsd/build_from_gateway.py: hinted tasks from a deployment's gateway spans, with no AC2 calls."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_builder():
    sys.path.insert(0, str(ROOT / "opsd"))
    spec = importlib.util.spec_from_file_location("build_from_gateway", ROOT / "opsd" / "build_from_gateway.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SYSTEM = "You are a professional poker player in a six-seat no-limit Texas Hold'em game against other AI players."


def row(span: str, call: str, arguments: str, system: str = SYSTEM, tokens_out: int = 400) -> dict:
    prompt = [
        {"type": "message", "role": "system", "content": system},
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
