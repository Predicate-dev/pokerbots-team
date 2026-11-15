"""Build an OPSD dataset from a deployment's gateway traces: one `HintedTask` per poker decision.

`build_dataset.py` needs an AC2 eval's traces. This reads the same thing from the inference gateway instead: every
request a deployment answered is recorded there with its prompt and reply, whoever sent it. So decisions the base
deployment made in arena games and in local games (`harness/play_local.py`, `harness/score_local.py`) can be used
without waiting for an eval job.

    uv run python opsd/build_from_gateway.py --deployment-id infer_xxx --dataset pokerbots-opsd-tight            # dry run
    uv run python opsd/build_from_gateway.py --deployment-id infer_xxx --dataset pokerbots-opsd-tight --upload

Poker turns must use the unmodified official system prompt. The builder keeps single action calls below the
token cap. `--table-talk-defense` and `--opponent-reads` also keep clean scratchpad turns. Use traces of the
training base model for offline OPSD because it replays the model's generated tokens. One-step tasks store
prompt prefixes so the student can sample its own action.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from ac2.sdk import Client

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import add_strategy_args, decision_tasks, selected_hints  # noqa: E402

PAGE = 200
COLUMNS = ["trace_id", "span_id", "status", "start_time", "tokens_out", "input", "output"]


def fetch_spans(client: Client, deployment_id: str, project: str, since: str | None) -> list[dict]:
    """Every completed gateway span of one deployment, oldest first."""
    from ac2.sdk.analytics.types import ColumnSpec, FieldFilter, QuerySpec

    filters = [
        FieldFilter(field="job_id", op="eq", value=deployment_id),
        FieldFilter(field="status", op="eq", value="completed"),
    ]
    if since:
        filters.append(FieldFilter(field="start_time", op="ge", value=since))
    rows: list[dict] = []
    while True:
        spec = QuerySpec(
            source="spans",
            projects=[project],
            columns=[ColumnSpec(field=name) for name in COLUMNS],
            filters=filters,
            sort_by="start_time",
            sort_order="asc",
            limit=PAGE,
            offset=len(rows),
            include_total=False,
        )
        page = client.analytics.query(spec=spec).rows
        rows.extend(page)
        if len(page) < PAGE:
            return rows


def as_trace(row: dict) -> dict:
    """One gateway span in the shape `build_dataset.decision_tasks` reads."""
    return {
        "trace_id": f"{row['trace_id']}:{row['span_id']}",
        "spans": [
            {
                "name": "get_completion",
                "status": "OK",
                "start_time": row["start_time"],
                "input": row["input"],
                "output": row["output"],
                "tokens_out": row.get("tokens_out") or 0,
            }
        ],
    }


def build(
    rows: list[dict], hint: str, max_tokens: int, *, cleanup_hint: str | None = None, cleanup_max_tokens: int = 10_000,
    rollout_mode: str = "offline",
) -> tuple[list, dict[str, int]]:
    totals = {"spans": len(rows), "not_poker": 0, "decisions": 0, "kept": 0, "cleanup": 0, "format": 0, "truncated": 0}
    if cleanup_hint is not None:
        totals["cleanup_kept"] = 0
    tasks = []
    seen: set[str] = set()
    for row in rows:
        key = f"{row['trace_id']}:{row['span_id']}"
        if key in seen:
            continue
        seen.add(key)
        if not row.get("input") or not row.get("output"):
            totals["not_poker"] += 1
            continue
        try:
            built, stats = decision_tasks(
                as_trace(row), hint, max_tokens, cleanup_hint=cleanup_hint, cleanup_max_tokens=cleanup_max_tokens,
                rollout_mode=rollout_mode,
            )
        except ValueError:  # a span whose items don't validate
            totals["format"] += 1
            continue
        if stats.pop("not_official"):
            totals["not_poker"] += 1
            continue
        tasks.extend(built)
        for name, count in stats.items():
            totals[name] = totals.get(name, 0) + count
    return tasks, totals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default="vals-hackathon")
    parser.add_argument("--deployment-id", required=True, help="the base deployment's id, infer_...")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--since", help="only spans from this ISO time on (UTC), e.g. 2026-11-14T22:30:00")
    parser.add_argument("--max-tokens", type=int, default=5_000, help="replies this long count as truncated")
    parser.add_argument("--cleanup-max-tokens", type=int, default=10_000, help="output cap for scratchpad updates")
    parser.add_argument("--opponent-reads", action="store_true", help="include clean cleanup turns and seat-read guidance")
    parser.add_argument("--table-talk-defense", action="store_true",
                        help="teach the model to ignore chatter when acting and updating notes")
    add_strategy_args(parser)
    parser.add_argument("--upload", action="store_true", help="create the dataset on AC2 (default: print only)")
    args = parser.parse_args()
    if args.max_tokens < 1 or args.cleanup_max_tokens < 1:
        parser.error("token caps must be positive")
    hint, cleanup_hint = selected_hints(args, parser)

    client = Client(project=args.project)
    rows = fetch_spans(client, args.deployment_id, args.project, args.since)
    tasks, totals = build(
        rows, hint, args.max_tokens, cleanup_hint=cleanup_hint, cleanup_max_tokens=args.cleanup_max_tokens,
        rollout_mode=args.rollout_mode,
    )
    random.Random(0).shuffle(tasks)
    print(json.dumps(totals | {"tasks": len(tasks)}))
    if not args.upload or not tasks:
        print("dry run: nothing uploaded (pass --upload to create the dataset)")
        return
    client.datasets.create(args.dataset, required_types=["hinted"], exist_ok=True)
    for start in range(0, len(tasks), 500):
        client.datasets.add_tasks(args.dataset, tasks=tasks[start : start + 500])
    print(f"uploaded {len(tasks)} tasks to {args.dataset}")


if __name__ == "__main__":
    main()
