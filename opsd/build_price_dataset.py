"""Build the price-card OPSD dataset from teacher replies recorded by the AC2 gateway.

    uv run python opsd/price_run.py --model vals-hackathon/<teacher> --spots harness/runs/price_spots.jsonl --hint --balance
    uv run python opsd/build_price_dataset.py --deployment-id infer_xxx --since 2026-11-15T06:00:00 --dataset pokerbots-opsd-price            # dry run
    uv run python opsd/build_price_dataset.py --deployment-id infer_xxx --since 2026-11-15T06:00:00 --dataset pokerbots-opsd-price --upload

`price_run.py --hint` asks the teacher each call decision with that decision's price card appended to the system
prompt. The gateway records the request and the reply. Each kept reply becomes one `HintedTask`: the official prompt
(card removed) followed by the teacher's reply, with the card as the teacher-only hint. Training then pulls the
student, which never sees the card, toward a reply that works the price out for itself.

Two kinds of reply are used. A plain reply is what the model wrote for the official prompt (`price_run.py` without
`--hint`): it is kept whatever it chose, as in the first OPSD run. A hinted reply is kept only if its action matches
the card's plan and its text does not talk about having been given anything. Both must be one valid `submit_action`
call under the token cap. Decisions from held-out matches are left out, and within each kind folds and continues are
kept in equal numbers.

Use `--decision open --spots harness/runs/open_spots.jsonl` for replies collected with the open card.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

from ac2.runtime import FunctionCall, HintedTask, HintMutation, HintSelection, Message
from ac2.sdk import Client

sys.path.insert(0, str(Path(__file__).resolve().parent))
import open_card  # noqa: E402
from build_dataset import ITEMS, SUBMIT, TOOLS, official_system_prompt  # noqa: E402
from build_from_gateway import fetch_spans  # noqa: E402
from hint import SEPARATOR  # noqa: E402
from price_card import card, hint  # noqa: E402

MARK = "Your read of this decision."
LEAK = re.compile(
    r"private numbers|i was (given|told|provided)|i'm told|(the|these|my) (hint|note|notes|numbers|read|info|estimates?) "
    r"(say|says|said|given|provided|tell|tells|is that|states?)|according to the|as (given|provided|instructed|stated)|"
    r"the (provided|given|stated) |\bhint\b|never say|instruction|the prompt|the user|the system|the info\b|the read\b|"
    r"your read|do not mention|being given",
    re.IGNORECASE,
)


def spot_key(c: dict) -> tuple:
    key = (c["hand"], tuple(c["hole"]), tuple(c["board"]), c["pot"], c["to_call"])
    return key + (c["position"], c["live"]) if "position" in c else key


def task_from_span(
    row: dict, max_tokens: int, held_out: set[tuple], decision: str = "call",
) -> tuple[HintedTask | None, str, str, str]:
    """(task, why it was dropped or "", the card's plan, "plain" or "hinted")."""
    try:
        prompt = ITEMS.validate_json(row["input"])
        reply = ITEMS.validate_json(row["output"])
    except ValueError:
        return None, "unreadable", "", ""
    official = official_system_prompt()
    first = prompt[0] if prompt else None
    if not isinstance(first, Message) or first.role != "system" or not isinstance(first.content, str):
        return None, "not a decision", "", ""
    opening = decision == "open"
    make_card = open_card.card if opening else card
    card_hint = open_card.hint if opening else hint
    marker = open_card.MARK if opening else MARK
    if first.content == official:
        kind = "plain"  # the model's own reply to the official prompt
    elif first.content.startswith(official + SEPARATOR + marker):
        kind = "hinted"  # the reply it wrote with the card in front of it
        prompt = [first.model_copy(update={"content": official}), *prompt[1:]]
    else:
        return None, "not a decision", "", ""
    c = make_card(prompt)
    if c is None:
        return None, f"not a {decision} decision", "", kind
    if spot_key(c) in held_out:
        return None, "held out", "", kind
    if not c.get("clear", c["plan"] != "close"):
        return None, "no clear plan", "", kind
    guidance = card_hint(c)
    if opening and kind == "hinted" and first.content != official + SEPARATOR + guidance:
        return None, "open card changed since the request", c["plan"], kind
    calls = [item for item in reply if isinstance(item, FunctionCall)]
    if len(calls) != 1 or calls[0].name != SUBMIT:
        return None, "format", c["plan"], kind
    if int(row.get("tokens_out") or 0) >= max_tokens:
        return None, "truncated", c["plan"], kind
    try:
        args = json.loads(calls[0].arguments)
        action = args.get("action")
    except (TypeError, ValueError, AttributeError):
        return None, "format", c["plan"], kind
    if opening:
        answer = open_card.answer(c, args)
        if answer == "invalid":
            return None, "opening action outside the card scope", c["plan"], kind
    else:
        answer = "fold" if action == "fold" else "continue" if action in ("call", "raise") else "invalid"
    if kind == "hinted":
        if answer != c["plan"]:
            return None, f"teacher chose {answer} against the card", c["plan"], kind
        if LEAK.search(row["output"]):
            return None, "talks about being given numbers", c["plan"], kind
    task = HintedTask(
        input=prompt + reply,
        env_params={"tools": TOOLS, "source_trace": f"{row['trace_id']}:{row['span_id']}"},
        opsd_hint_selections=[
            HintSelection(
                hint=guidance,
                rollout_prefix_len=len(prompt),
                mutation=HintMutation(item_boundary=1, action="append_to_previous_item", separator=SEPARATOR),
            )
        ],
    )
    return task, "", c["plan"], kind


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default="vals-hackathon")
    parser.add_argument("--deployment-id", required=True, help="the teacher deployment's id, infer_...")
    parser.add_argument("--since", required=True, help="only spans from this ISO time on (UTC)")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--decision", choices=("call", "open"), default="call")
    parser.add_argument("--spots", type=Path, required=True, help="replay spots, to leave out held-out matches")
    parser.add_argument("--max-tokens", type=int, default=5_000)
    parser.add_argument("--upload", action="store_true", help="create the dataset on AC2 (default: print only)")
    parser.add_argument("--sample", type=int, default=0, help="print this many kept replies")
    args = parser.parse_args()
    other = "raise" if args.decision == "open" else "continue"
    make_card = open_card.card if args.decision == "open" else card
    held_out = set()
    for line in args.spots.read_text().splitlines():
        spot = json.loads(line)
        if spot["held_out"] and (c := make_card(spot["messages"])) is not None:
            held_out.add(spot_key(c))
    client = Client(project=args.project)
    rows = fetch_spans(client, args.deployment_id, args.project, args.since)
    stats: Counter = Counter()
    kept: dict[tuple[str, str], list[HintedTask]] = {(k, p): [] for k in ("plain", "hinted") for p in ("fold", other)}
    seen: set[str] = set()
    for row in rows:
        if not row.get("input") or not row.get("output"):
            continue
        task, problem, plan, kind = task_from_span(row, args.max_tokens, held_out, args.decision)
        if problem:
            stats[f"{kind}: {problem}" if kind else problem] += 1
            continue
        n = task.opsd_hint_selections[0].rollout_prefix_len
        key = kind + json.dumps([getattr(item, "content", None) for item in task.input[1:n]])
        if key in seen:
            stats[f"{kind}: duplicate"] += 1
            continue
        seen.add(key)
        kept[(kind, plan)].append(task)
    rng = random.Random(0)
    tasks = []
    for kind in ("plain", "hinted"):
        size = min(len(kept[(kind, "fold")]), len(kept[(kind, other)]))
        tasks += rng.sample(kept[(kind, "fold")], size) + rng.sample(kept[(kind, other)], size)
    rng.shuffle(tasks)
    print(json.dumps({"spans": len(rows), "tasks": len(tasks)} | {f"kept {k} {p}": len(v) for (k, p), v in kept.items()} | dict(stats)))
    for task in tasks[: args.sample]:
        n = task.opsd_hint_selections[0].rollout_prefix_len
        print("\n--- hint ---\n" + task.opsd_hint_selections[0].hint)
        for item in task.input[n:]:
            print(f"--- {type(item).__name__} ---\n{getattr(item, 'content', None) or getattr(item, 'arguments', '')}")
    if not args.upload or not tasks:
        return
    client.datasets.create(args.dataset, required_types=["hinted"], exist_ok=True)
    for start in range(0, len(tasks), 500):
        client.datasets.add_tasks(args.dataset, tasks=tasks[start : start + 500])
    print(f"uploaded {len(tasks)} tasks to {args.dataset}")


if __name__ == "__main__":
    main()
