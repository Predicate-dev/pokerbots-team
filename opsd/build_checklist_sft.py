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
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness" / "src"))
import open_card  # noqa: E402
import playbook_rules  # noqa: E402
from build_dataset import ITEMS, SCRATCHPAD, SUBMIT, TOOLS, is_official_prompt  # noqa: E402
from build_from_gateway import fetch_spans  # noqa: E402
from build_price_dataset import spot_key  # noqa: E402
from poker_harness.opponent_reads import betting_evidence  # noqa: E402
from poker_harness.prompts import parse_tool_call, prompt_legal_actions  # noqa: E402
from poker_harness.seat import invalid_action  # noqa: E402
from price_card import (  # noqa: E402
    POSTFLOP_NAMES,
    _strength,
    card,
    parse_state,
    postflop_class,
    preflop_class,
)
from price_run import PAD, with_notes  # noqa: E402

NOTES = Path(__file__).with_name("notes") / "call_checklist.txt"
KEEP = "The CALL CHECKLIST block stays at the top of my notes, word for word. Below it I write my notes for this hand."
KEEP_BOTH = ("The CALL CHECKLIST and OPPONENT GUIDE blocks stay at the top of my notes, word for word. "
             "Below them I update each seat's counters from the betting actions I observed.")
KEEP_PLAYBOOK = ("The CALL CHECKLIST and PLAYBOOK blocks stay at the top of my notes, word for word. Below them I write "
                 "my notes for this hand.")
LIMIT = 5_000
BIG_SHARE = 0.4  # a call for this share of the stack or more is a "big call"
NO_NOTES_RULE = ("My scratchpad is empty. Estimate the opponent's range from position and betting. Compare the call "
                 "price and the effect of winning or losing on my finishing place.")
REPLACED = ""  # an earlier wording of the block that generated replies were shown (`--generated-with`)
DROPPED_LINE_USE = re.compile(r"subtract 20|show me|oh no, sorry|never bets without|\b(?:rule|playbook) [DE]\b|\b[DE]\)", re.IGNORECASE)
# a call decision is a training target only if its reply works from the checklist, not merely agrees with it
USES_NOTES = re.compile(r"checklist|place margin|win ?%|\bneed\b|opponent guide|my notes", re.IGNORECASE)
_OLD_BLOCKS = (
    re.compile(r"CALL CHECKLIST.*?(?=\n\n|\Z)", re.DOTALL),  # a block has no blank line inside it
    re.compile(r"OPPONENT GUIDE.*?(?=\n\n|\Z)", re.DOTALL),
    re.compile(r"PLAYBOOK.*?(?=\n\n|\Z)", re.DOTALL),
    re.compile(r"^Seat types:[^\n]*\n?", re.MULTILINE),
)


def strip_blocks(text: str) -> str:
    """Notes without any earlier copy of the reference blocks or the seat-types line (they are written fresh)."""
    for pattern in _OLD_BLOCKS:
        text = pattern.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def clean_pad(pad_message: str) -> str:
    """The scratchpad message with earlier copies of the blocks removed from the scratchpad it shows."""
    found = PAD.search(pad_message)
    if not found or found[1].strip() == "(empty)":
        return pad_message
    body = strip_blocks(found[1]) or "(empty)"
    size = 0 if body == "(empty)" else len(body)
    return pad_message[: found.start()] + f"<<<SCRATCHPAD\n{body}\nSCRATCHPAD>>>\n{size}/{found[3]} characters" + pad_message[found.end():]


def hand_name(hole: list[str], board: list[str]) -> str:
    made = postflop_class(hole, board)
    if made == "monster":
        return {4: "a straight", 5: "a flush"}.get(_strength(hole + board), "a full house or better")
    return POSTFLOP_NAMES[made]


def lead_reasoning(reply: list[dict], line: str) -> None:
    for item in reply:
        if item.get("type") == "reasoning" and isinstance(item.get("content"), str):
            item["content"] = f"{line}\n\n{item['content']}"
            return
    reply.insert(0, {"type": "reasoning", "content": line})


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
    contents = strip_blocks(contents)  # a trace from a model that already kept a block: write the block fresh
    empty = "(empty)" in prompt[1]["content"]
    prompt[1] = prompt[1] | {"content": clean_pad(prompt[1]["content"])}
    if show_block and not empty:
        prompt[1] = prompt[1] | {"content": with_notes(prompt[1]["content"], block)}
    guide = "OPPONENT GUIDE" in block
    keep = KEEP_BOTH if guide else KEEP_PLAYBOOK if "PLAYBOOK" in block else KEEP
    if guide:
        messages = [m["content"] for m in prompt[2:] if m.get("role") == "user" and isinstance(m.get("content"), str)]
        contents = betting_evidence(messages, prompt[1]["content"])
    new = f"{block}\n\n{contents}"[:LIMIT]
    call["arguments"] = json.dumps({"contents": new})
    for item in reply:  # a reasoning that already opens with the keep sentence gets it once
        if item.get("type") == "reasoning" and isinstance(item.get("content"), str):
            item["content"] = "" if guide else item["content"].replace(KEEP_PLAYBOOK, "").replace(KEEP, "").strip()
    lead_reasoning(reply, keep)
    kind = "first" if empty else "keep" if show_block else "restore"
    return close(prompt, reply, call, "Scratchpad updated.", f"checklist:notes:{kind}")


def decision_task(row: dict, block: str, held_out: set, max_tokens: int, need_block: bool,
                  extra_pools: set[str] | None = None) -> tuple[Task | None, str]:
    """A decision with the blocks in the scratchpad it was shown; (task, kind or why it was dropped).

    Kinds: call:fold / call:continue (a clear price-card plan, followed), open:raise / open:fold (a clear open-card
    plan, followed), premium (a raise with QQ+, AK, 99-JJ or AQ-AJ before the flop), other.

    With a PLAYBOOK block the playbook's own rules replace the open card (`playbook_rules.verdicts`): a reply that
    breaks a rule is dropped, and one that follows a rule is kind open:raise / open:fold (rule A), flop bet (B),
    reraise / reraise answer (D). Call decisions from real play are then kept when the reply works from the notes."""
    found = parts(row)
    if found is None:
        return None, "not clean"
    prompt, reply, call, arguments = found
    if call["name"] != SUBMIT or int(row.get("tokens_out") or 0) >= max_tokens:
        return None, "not a short action"
    action = arguments.get("action")
    playbook = "PLAYBOOK" in block
    rules = playbook_rules.verdicts(prompt, arguments) if playbook else []
    if any(not followed for _, followed in rules):
        return None, "against the playbook: " + ",".join(rule for rule, followed in rules if not followed)
    if arguments.get("say") and playbook:
        return None, "talks"
    if (REPLACED and "PLAYBOOK" in prompt[1]["content"] and block not in prompt[1]["content"]
            and DROPPED_LINE_USE.search(json.dumps(reply))):
        return None, "leans on a removed line"
    if need_block and REPLACED and REPLACED in prompt[1]["content"]:
        # the reply was written with an earlier wording of the block: show the current one, and drop replies that
        # lean on a line the current block no longer has
        if DROPPED_LINE_USE.search(json.dumps(reply)):
            return None, "leans on a removed line"
        prompt[1] = prompt[1] | {"content": with_notes(clean_pad(prompt[1]["content"]), block)}
    has_block = block in prompt[1]["content"]
    if need_block and not has_block:
        return None, "no checklist in the notes"
    state = parse_state(prompt)
    if state is None:
        return None, "unreadable state"
    spot, rest = state
    decision, invalid = parse_tool_call(arguments)
    legal = prompt_legal_actions(rest)
    if invalid or legal is None or invalid_action(decision, legal):
        return None, "illegal action"
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
        big = c["share"] >= BIG_SHARE and kind != "other"
        strong = (spot["street"] != "preflop" and kind == "call:continue"
                  and postflop_class(spot["hole"], spot["board"]) in ("monster", "trips", "two_pair", "overpair"))
        big_kind = "big fold" if answer == "fold" else "big continue"
        if strong and extra_pools and "strong continue" in extra_pools:
            kind = "strong continue"
        elif big and extra_pools and big_kind in extra_pools:
            if spot["hand"] == 1 and spot["street"] == "preflop" and "(empty)" in prompt[1]["content"]:
                lead_reasoning(reply, NO_NOTES_RULE)
            kind = big_kind
        elif not need_block and kind != "other":  # real play: only the first hand, where the notes are still empty
            premium_raise = (spot["street"] == "preflop" and action == "raise"
                             and preflop_class(spot["hole"]) in ("QQ+", "AK", "99-JJ", "AQ-AJ"))
            if premium_raise:
                kind = "other"  # becomes "premium" below
            elif spot["hand"] == 1 and "(empty)" in prompt[1]["content"]:
                kind = "first-hand call"
            elif "CALL CHECKLIST" in prompt[1]["content"] and USES_NOTES.search(json.dumps(reply)):
                pass  # the model already plays with the checklist: its own call decisions are targets
            else:
                return None, "a call decision from real play after the first hand"
    if spot["street"] == "preflop":
        hand = preflop_class(spot["hole"])
        can_raise = "raise to" in rest
        opening = None if playbook else open_card.card(prompt)
        for rule, _ in rules:
            if rule == "A":
                kind = "open:raise" if action == "raise" else "open:fold"
            elif rule == "D":
                kind = "reraise" if action == "raise" else "reraise answer"
        if opening is not None:  # everyone has folded to this seat
            if action == "call":
                return None, "limps"
            if opening["clear"]:
                if open_card.answer(opening, arguments) != opening["plan"]:
                    return None, "against the open card"
                kind = f"open:{opening['plan']}"
        if hand in ("QQ+", "AK") and can_raise and action in ("fold", "call") and not (c and c["plan"] == "fold"):
            return None, "does not raise a premium hand"
        if hand in ("QQ+", "AK", "99-JJ", "AQ-AJ") and action == "raise" and kind == "other":
            kind = "premium"
    else:
        if ("B", True) in rules:
            kind = "flop bet"
        made = postflop_class(spot["hole"], spot["board"])
        if made in ("monster", "trips") and action == "fold" and (c is None or c["plan"] != "fold"):
            return None, "folds a very strong hand without support from the card"
        lead_reasoning(reply, f"My hand: {hand_name(spot['hole'], spot['board'])}.")  # name the made hand before anything else
    if not has_block:  # a trace from real play: show the notes as this model will have them
        pad = clean_pad(prompt[1]["content"])
        prompt[1] = prompt[1] | {"content": pad if "(empty)" in pad and spot["hand"] == 1 else with_notes(pad, block)}
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
    parser.add_argument("--policy-deployment-id", required=True, nargs="+", help="the model's deployments in real games, infer_...")
    parser.add_argument("--since", required=True, nargs="+", help="for each policy deployment: spans from this ISO time on (UTC)")
    parser.add_argument("--notes-deployment-id", help="the deployment asked call decisions with --notes-file; without it "
                        "(playbook runs) the call decisions come from the policy deployments' own play")
    parser.add_argument("--notes-since", help="notes spans from this ISO time on (UTC)")
    parser.add_argument("--call-decisions", type=int, default=300, help="call decisions of each answer, when taken from play")
    parser.add_argument("--big-folds", type=int, default=0, help="folds to a call for 40%+ of the stack that the card says to fold")
    parser.add_argument("--strong-continues", type=int, default=0,
                        help="continues after the flop with two pair or better or an overpair, when the card says continue")
    parser.add_argument("--big-fold-copies", type=int, default=1, help="times each big fold or big continue appears")
    parser.add_argument("--big-continues", type=int, default=0, help="calls for 40%+ of the stack that the card says to make")
    parser.add_argument("--flop-bets", type=int, default=200, help="playbook rule B: flop bets after raising")
    parser.add_argument("--reraises", type=int, default=150, help="playbook rule D: re-raises of an open")
    parser.add_argument("--reraise-answers", type=int, default=100, help="playbook rule D: answers to a re-raise of our raise")
    parser.add_argument("--spots", type=Path, required=True, help="harness/runs/price_spots.jsonl, to leave out held-out matches")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--notes-file", type=Path, default=NOTES, help="the text to keep at the top of the scratchpad")
    parser.add_argument("--generated-with", type=Path, help="an earlier wording of the notes that the generated replies were shown")
    parser.add_argument("--notes-turns", type=int, default=1_200)
    parser.add_argument("--restore-share", type=float, default=0.45, help="share of later notes turns shown without the block")
    parser.add_argument("--other-decisions", type=int, default=500)
    parser.add_argument("--open-raises", type=int, default=300, help="first-in raises the open card agrees with")
    parser.add_argument("--open-folds", type=int, default=200, help="first-in folds the open card agrees with")
    parser.add_argument("--premium-raises", type=int, default=300, help="preflop raises with QQ+, AK, 99-JJ, AQ-AJ")
    parser.add_argument("--first-hand-calls", type=int, default=300, help="hand-1 call decisions (empty notes) that follow the card")
    parser.add_argument("--max-tokens", type=int, default=2_500, help="drop longer replies")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    block = args.notes_file.read_text().strip()
    global REPLACED  # noqa: PLW0603
    REPLACED = args.generated_with.read_text().strip() if args.generated_with else ""
    held_out = set()
    for line in args.spots.read_text().splitlines():
        spot = json.loads(line)
        if spot["held_out"] and (c := card(spot["messages"])) is not None:
            held_out.add(spot_key(c))
    client = Client(project=args.project)
    rng = random.Random(0)
    stats: Counter = Counter()

    policy = []
    for deployment, since in zip(args.policy_deployment_id, args.since, strict=True):
        policy += fetch_spans(client, deployment, args.project, since)
    rng.shuffle(policy)
    first, later = [], []
    pools: dict[str, list[Task]] = {"other": [], "open:raise": [], "open:fold": [], "premium": [], "first-hand call": [],
                                    "flop bet": [], "reraise": [], "reraise answer": [], "big fold": [], "big continue": [], "strong continue": []}
    caps = {"other": args.other_decisions, "open:raise": args.open_raises, "open:fold": args.open_folds, "premium": args.premium_raises,
            "first-hand call": args.first_hand_calls, "flop bet": args.flop_bets, "reraise": args.reraises,
            "reraise answer": args.reraise_answers, "big fold": args.big_folds, "big continue": args.big_continues, "strong continue": args.strong_continues}
    extra_pools = {kind for kind in ("big fold", "big continue", "strong continue") if caps[kind] > 0}
    calls: dict[str, list[Task]] = {"call:fold": [], "call:continue": []}
    for row in policy:
        if '"update_scratchpad"' in (row.get("output") or ""):
            show = rng.random() >= args.restore_share
            task = notes_task(row, block, show, args.max_tokens)
            if task is not None:
                (first if task.env_params["source"].endswith("first") else later).append(task)
        else:
            task, kind = decision_task(row, block, held_out, args.max_tokens, need_block=False, extra_pools=extra_pools)
            if task is None:
                stats[f"ordinary decision dropped: {kind}"] += 1
            elif kind in pools:
                pools[kind].append(task)
            elif kind in calls and not args.notes_deployment_id:
                calls[kind].append(task)
    other = [task for kind, tasks in pools.items() for task in tasks[: caps[kind]]]
    other += (pools["big fold"][: caps["big fold"]] + pools["big continue"][: caps["big continue"]]) * (args.big_fold_copies - 1)  # the examples that stop busts, weighted up
    for kind, tasks in pools.items():
        stats[f"kept {kind}"] = min(len(tasks), caps[kind])
    notes_turns = first + later[: max(0, args.notes_turns - len(first))]
    for task in notes_turns:
        stats[task.env_params["source"]] += 1

    for row in fetch_spans(client, args.notes_deployment_id, args.project, args.notes_since) if args.notes_deployment_id else []:
        task, kind = decision_task(row, block, held_out, args.max_tokens, need_block=True)
        if task is None:
            stats[f"call decision dropped: {kind}"] += 1
        elif kind in calls:
            calls[kind].append(task)
    size = min(len(v) for v in calls.values())
    if not args.notes_deployment_id:
        size = min(size, args.call_decisions)
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
