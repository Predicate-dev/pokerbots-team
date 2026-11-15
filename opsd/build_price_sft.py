"""Build a supervised (SFT) dataset of price-card decisions from replies the AC2 gateway recorded.

    uv run python opsd/build_price_sft.py --deployment-id infer_xxx --since 2026-11-15T06:00:00 \
        --spots harness/runs/price_spots.jsonl --dataset pokerbots-sft-price            # dry run; add --upload
    uv run python sft/train.py --dataset pokerbots-sft-price --from-checkpoint <ckpt> --steps 15 --checkpoint-every 5

The targets are the model's own replies, in its own format, in two equal halves:

- hinted: replies written with the decision's price card in front of the model (`price_run.py --hint`) that follow
  the card, never mention being given anything, and are short. These spell the reasoning out: the chip price, what
  losing and winning do to the finishing place, how often the hand wins, the comparison.
- plain: replies to the official prompt (`price_run.py` without `--hint`) whose action already matches the card.
  These keep the model's normal voice and length.

The card itself is not in the conversation: the student sees the official prompt only. Each target ends with the
tool's reply, which closes the turn (see `sft/build_dataset.py`). Decisions from held-out matches are left out, and
within each half folds and continues are kept in equal numbers.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from ac2.runtime import FunctionCall, Task
from ac2.sdk import Client

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import ITEMS, TOOLS  # noqa: E402
from build_from_gateway import fetch_spans  # noqa: E402
from build_price_dataset import spot_key, task_from_span  # noqa: E402
from price_card import card  # noqa: E402


def supervised(hinted_task, c: dict, index: int) -> tuple[Task, str]:
    """The official prompt and the reply as one conversation, closed by the tool's reply; and the reply's answer."""
    items = list(hinted_task.input)
    call = next(item for item in reversed(items) if isinstance(item, FunctionCall))
    arguments = json.loads(call.arguments)
    action = arguments.get("action")
    move = "fold" if action == "fold" else f"call {min(c['to_call'], c['stack'])}" if action == "call" else f"raise to {arguments.get('amount')}"
    conversation = json.loads(ITEMS.dump_json(items)) + [
        {"type": "function_call_output", "call_id": call.call_id, "output": f"Applied: {move}."}
    ]
    answer = "fold" if action == "fold" else "continue" if action in ("call", "raise") else "invalid"
    return Task(input=ITEMS.validate_json(json.dumps(conversation)), env_params={"tools": TOOLS, "source": f"price:{index}"}), answer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default="vals-hackathon")
    parser.add_argument("--deployment-id", required=True, help="the deployment that wrote the replies, infer_...")
    parser.add_argument("--since", required=True, help="only spans from this ISO time on (UTC)")
    parser.add_argument("--spots", type=Path, required=True, help="harness/runs/price_spots.jsonl, to leave out held-out matches")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--max-tokens", type=int, default=2_000, help="drop replies longer than this: long ones ramble")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    held_out = set()
    for line in args.spots.read_text().splitlines():
        spot = json.loads(line)
        if spot["held_out"] and (c := card(spot["messages"])) is not None:
            held_out.add(spot_key(c))
    client = Client(project=args.project)
    rows = fetch_spans(client, args.deployment_id, args.project, args.since)
    stats: Counter = Counter()
    kept: dict[tuple[str, str], list[Task]] = {(k, p): [] for k in ("plain", "hinted") for p in ("fold", "continue")}
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not row.get("input") or not row.get("output"):
            continue
        task, problem, plan, kind = task_from_span(row, args.max_tokens, held_out)
        if problem:
            stats[f"{kind}: {problem}" if kind else problem] += 1
            continue
        n = task.opsd_hint_selections[0].rollout_prefix_len
        c = card(task.input[:n])
        key = kind + json.dumps([getattr(item, "content", None) for item in task.input[1:n]])
        if key in seen:
            stats[f"{kind}: duplicate"] += 1
            continue
        seen.add(key)
        target, answer = supervised(task, c, index)
        if answer != plan:  # plain replies are kept only where the model already did what the card says
            stats[f"{kind}: chose {answer} against the card"] += 1
            continue
        kept[(kind, plan)].append(target)
    rng = random.Random(0)
    size = min(len(v) for v in kept.values())
    tasks = [t for v in kept.values() for t in rng.sample(v, size)]
    rng.shuffle(tasks)
    print(json.dumps({"spans": len(rows), "tasks": len(tasks), "per kind and plan": size}
                     | {f"kept {k} {p}": len(v) for (k, p), v in kept.items()} | dict(stats)))
    if not args.upload or not tasks:
        return
    client.datasets.create(args.dataset, exist_ok=True)
    for start in range(0, len(tasks), 500):
        client.datasets.add_tasks(args.dataset, tasks=tasks[start : start + 500])
    print(f"uploaded {len(tasks)} tasks to {args.dataset}")


if __name__ == "__main__":
    main()
