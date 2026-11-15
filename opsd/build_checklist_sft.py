"""Build a supervised dataset that teaches the model to keep a call checklist at the top of its scratchpad.

    uv run python opsd/build_checklist_sft.py --policy-deployment-id infer_xxx --notes-deployment-id infer_yyy \
        --since 2026-11-15T00:00:00 --notes-since 2026-11-15T07:40:00 --spots harness/runs/price_spots.jsonl \
        --dataset pokerbots-sft-checklist            # dry run; add --upload
    uv run python sft/train.py --dataset pokerbots-sft-checklist --from-checkpoint <ckpt>

With the checklist (`opsd/notes/call_checklist.txt`) already in its scratchpad, the unchanged model answers held-out
call decisions much better (`price_run.py --notes-file`). The arena starts every game with an empty scratchpad, so
the model has to write the block itself after its first hand and keep it there. This dataset teaches that, using
the model's own recorded turns as targets:

- notes turns (`update_scratchpad`): the model's real updates, with the checklist as the first block of the new
  contents. Where the scratchpad shown at the start of the hand was empty, or in a share of the other examples where
  the block is left out of it, the model has to write the block from memory; in the rest it is already there and
  must be kept word for word.
- call decisions with the checklist in the notes: replies the model wrote with the block in its scratchpad
  (`price_run.py --notes-file`) whose action matches the price card.
- other decisions: the model's own ordinary replies, with the block added to the scratchpad it was shown, so play
  that the checklist says nothing about stays as it was.

Every target ends with the tool's reply, which closes the turn. Decisions from held-out matches are left out.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

from ac2.runtime import Task
from ac2.sdk import Client

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import ITEMS, SCRATCHPAD, SUBMIT, TOOLS, is_official_prompt  # noqa: E402
from build_from_gateway import fetch_spans  # noqa: E402
from build_price_dataset import spot_key  # noqa: E402
from price_card import card  # noqa: E402
from price_run import with_notes  # noqa: E402

NOTES = Path(__file__).with_name("notes") / "call_checklist.txt"
KEEP = "The CALL CHECKLIST block stays at the top of my notes, word for word. Below it I write my notes for this hand."
KEEP_BOTH = ("The CALL CHECKLIST and OPPONENT GUIDE blocks stay at the top of my notes, word for word. Below them I name "
             "each seat's type from its talk and play, then write my notes for this hand.")
LIMIT = 5_000
# a call decision is a training target only if its reply works from the checklist, not merely agrees with it
USES_NOTES = re.compile(r"checklist|place margin|win ?%|\bneed\b|opponent guide|my notes", re.IGNORECASE)
_SAID = re.compile(r'\(seat (\d+)\)[^\n]*? says: "([^"\n]*)"')
_HERO = re.compile(r"Your seat: (\d+)")
TYPES = (  # the guide's types, by the lines that give them away
    ("NEVER FOLDS", ("i call, show me", "show me", "you are bluffing", "not buying it")),
    ("SCARED", ("oh no", "please do not raise me")),
    ("ALL-IN OR FOLD", ("all in, baby", "all in baby")),
    ("TALKER", ("flopped the nuts", "ugh,", "ugh ")),
    ("PROFESSOR", ("range", "discipline", "this holding", "implied odds")),
)


def seat_types(prompt: list[dict]) -> str:
    """"Seat types: seat 2 = SCARED; ..." from the talk shown in this hand, or "" when no line gives a type away."""
    text = "\n".join(str(m.get("content")) for m in prompt[2:] if m.get("role") == "user")
    hero = _HERO.search(text)
    found: dict[int, str] = {}
    for seat, line in _SAID.findall(text):
        if hero and seat == hero[1]:
            continue
        for name, phrases in TYPES:
            if any(phrase in line.lower() for phrase in phrases):
                found.setdefault(int(seat), name)
                break
    return "Seat types: " + "; ".join(f"seat {seat} = {name}" for seat, name in sorted(found.items())) if found else ""


def parts(row: dict) -> tuple[list[dict], list[dict], dict, dict] | None:
    """(prompt items, reply items, the reply's one tool call, its arguments) for a clean official-prompt span."""
    if not row.get("input") or not row.get("output") or not is_official_prompt(row["input"]):
        return None
    prompt, reply = json.loads(row["input"]), json.loads(row["output"])
    calls = [item for item in reply if item.get("type") == "function_call"]
    if len(calls) != 1 or len(prompt) < 3 or prompt[1].get("role") != "user" or "<<<SCRATCHPAD" not in str(prompt[1].get("content")):
        return None
    try:
        arguments = json.loads(calls[0]["arguments"])
    except (TypeError, ValueError):
        return None
    return (prompt, reply, calls[0], arguments) if isinstance(arguments, dict) else None


def close(prompt: list[dict], reply: list[dict], call: dict, output: str, source: str) -> Task:
    conversation = prompt + reply + [{"type": "function_call_output", "call_id": call["call_id"], "output": output}]
    return Task(input=ITEMS.validate_json(json.dumps(conversation)), env_params={"tools": TOOLS, "source": source})


def notes_task(row: dict, block: str, show_block: bool, max_tokens: int) -> Task | None:
    found = parts(row)
    if found is None:
        return None
    prompt, reply, call, arguments = found
    contents = arguments.get("contents")
    if call["name"] != SCRATCHPAD or not isinstance(contents, str) or int(row.get("tokens_out") or 0) >= max_tokens:
        return None
    if block in contents or "CALL CHECKLIST" in contents:
        return None
    empty = "(empty)" in prompt[1]["content"]
    if show_block and not empty:
        prompt[1] = prompt[1] | {"content": with_notes(prompt[1]["content"], block)}
    guide = "OPPONENT GUIDE" in block
    types = seat_types(prompt) if guide else ""
    new = f"{block}\n\n{types + chr(10) if types else ''}{contents.strip()}"[:LIMIT]
    call["arguments"] = json.dumps({"contents": new})
    keep = KEEP_BOTH if guide else KEEP
    for item in reply:
        if item.get("type") == "reasoning" and isinstance(item.get("content"), str):
            item["content"] = f"{keep}\n\n{item['content']}"
            break
    else:
        reply.insert(0, {"type": "reasoning", "content": keep})
    kind = "first" if empty else "keep" if show_block else "restore"
    return close(prompt, reply, call, "Scratchpad updated.", f"checklist:notes:{kind}")


def decision_task(row: dict, block: str, held_out: set, max_tokens: int, need_block: bool) -> tuple[Task | None, str]:
    """An ordinary or call decision with the checklist in the scratchpad it was shown; (task, kind or why dropped)."""
    found = parts(row)
    if found is None:
        return None, "not clean"
    prompt, reply, call, arguments = found
    if call["name"] != SUBMIT or int(row.get("tokens_out") or 0) >= max_tokens:
        return None, "not a short action"
    action = arguments.get("action")
    has_block = block in prompt[1]["content"]
    if need_block and not has_block:
        return None, "no checklist in the notes"
    c = card(prompt)
    kind = "other"
    if c is not None:
        if spot_key(c) in held_out:
            return None, "held out"
        answer = "fold" if action == "fold" else "continue" if action in ("call", "raise") else "invalid"
        if c["plan"] != "close" and answer != c["plan"]:
            return None, "against the card"
        kind = f"call:{answer}" if c["plan"] != "close" else "other"
        if need_block and kind != "other" and not USES_NOTES.search(json.dumps(reply)):
            return None, "does not work from the notes"
    if not has_block and "(empty)" not in prompt[1]["content"]:  # every later hand has the block in its notes
        prompt[1] = prompt[1] | {"content": with_notes(prompt[1]["content"], block)}
    if action == "fold":
        move = "fold"
    elif action == "check":
        move = "check"
    elif action == "call":
        move = f"call {min(c['to_call'], c['stack'])}" if c else "call"
    else:
        move = f"{action} to {arguments.get('amount')}"
    return close(prompt, reply, call, f"Applied: {move}.", f"checklist:{kind}"), kind


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default="vals-hackathon")
    parser.add_argument("--policy-deployment-id", required=True, help="the model's deployment in real games, infer_...")
    parser.add_argument("--since", required=True, help="policy spans from this ISO time on (UTC)")
    parser.add_argument("--notes-deployment-id", required=True, help="the deployment asked with --notes-file")
    parser.add_argument("--notes-since", required=True, help="notes spans from this ISO time on (UTC)")
    parser.add_argument("--spots", type=Path, required=True, help="harness/runs/price_spots.jsonl, to leave out held-out matches")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--notes-file", type=Path, default=NOTES, help="the text to keep at the top of the scratchpad")
    parser.add_argument("--notes-turns", type=int, default=1_200)
    parser.add_argument("--restore-share", type=float, default=0.45, help="share of later notes turns shown without the block")
    parser.add_argument("--other-decisions", type=int, default=500)
    parser.add_argument("--max-tokens", type=int, default=2_500, help="drop longer replies")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    block = args.notes_file.read_text().strip()
    held_out = set()
    for line in args.spots.read_text().splitlines():
        spot = json.loads(line)
        if spot["held_out"] and (c := card(spot["messages"])) is not None:
            held_out.add(spot_key(c))
    client = Client(project=args.project)
    rng = random.Random(0)
    stats: Counter = Counter()

    policy = fetch_spans(client, args.policy_deployment_id, args.project, args.since)
    rng.shuffle(policy)
    first, later, other = [], [], []
    for row in policy:
        if '"update_scratchpad"' in (row.get("output") or ""):
            show = rng.random() >= args.restore_share
            task = notes_task(row, block, show, args.max_tokens)
            if task is not None:
                (first if task.env_params["source"].endswith("first") else later).append(task)
        elif len(other) < args.other_decisions:
            task, kind = decision_task(row, block, held_out, args.max_tokens, need_block=False)
            if task is not None and kind == "other":
                other.append(task)
    notes_turns = first + later[: max(0, args.notes_turns - len(first))]
    for task in notes_turns:
        stats[task.env_params["source"]] += 1

    calls: dict[str, list[Task]] = {"call:fold": [], "call:continue": []}
    for row in fetch_spans(client, args.notes_deployment_id, args.project, args.notes_since):
        task, kind = decision_task(row, block, held_out, args.max_tokens, need_block=True)
        if task is None:
            stats[f"call decision dropped: {kind}"] += 1
        elif kind in calls:
            calls[kind].append(task)
    size = min(len(v) for v in calls.values())
    call_tasks = rng.sample(calls["call:fold"], size) + rng.sample(calls["call:continue"], size)

    tasks = notes_turns + call_tasks + other
    rng.shuffle(tasks)
    print(json.dumps({"tasks": len(tasks), "notes turns": len(notes_turns), "call decisions": len(call_tasks),
                      "other decisions": len(other), "kept call folds": len(calls["call:fold"]),
                      "kept call continues": len(calls["call:continue"])} | dict(stats)))
    if not args.upload or not tasks:
        return
    client.datasets.create(args.dataset, exist_ok=True)
    for start in range(0, len(tasks), 500):
        client.datasets.add_tasks(args.dataset, tasks=tasks[start : start + 500])
    print(f"uploaded {len(tasks)} tasks to {args.dataset}")


if __name__ == "__main__":
    main()
