"""Ask a deployment the call decisions in `price_spots.jsonl`, with or without the price card as a hint.

    # test a model on held-out matches (official prompt, no hint): rating points per 1000 decisions
    uv run python opsd/price_run.py --model vals-hackathon/<deployment> --spots harness/runs/price_spots.jsonl --held-out

    # write teacher replies for distillation: the same model, shown each decision's price card
    uv run python opsd/price_run.py --model vals-hackathon/<deployment> --spots harness/runs/price_spots.jsonl \
        --hint --balance --out harness/runs/price_teacher.jsonl

A reply is scored by the rating points its answer is worth over folding, from the spot's real equity and the stack
values in `price_tables.json` (see `price_card.py`). With `--hint` the card's text is appended to the system prompt,
exactly where OPSD puts a hint; the request is recorded by the AC2 gateway, which `build_price_dataset.py` reads.

Use `--decision open --spots harness/runs/open_spots.jsonl` to ask first-in raise-or-fold decisions. Opening replies
are reported by raise rate and agreement with the open card. Use the held-out replay report for opening values.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness" / "src"))
import open_card  # noqa: E402
from hint import SEPARATOR  # noqa: E402
from poker_harness.completions import GATEWAY_URL, relay_completion, with_preset  # noqa: E402
from poker_harness.opponent_reads import replay_betting_evidence  # noqa: E402
from poker_harness.prompts import (  # noqa: E402
    SUBMIT_ACTION,
    parse_tool_call,
    prompt_legal_actions,
    talk_fields,
)
from poker_harness.seat import Completion, invalid_action  # noqa: E402
from price_card import card, hint  # noqa: E402

PAD = re.compile(r"<<<SCRATCHPAD\n(.*)\nSCRATCHPAD>>>\n(\d+)/(\d+) characters", re.DOTALL)
SPOT_ID = re.compile(r"([A-Za-z0-9_-]+):([1-9]\d*):([0-5]):\d+")


def with_notes(pad_message: str, notes: str, *, replace: bool = False) -> str:
    """Put notes into the private message, respecting its character limit."""
    found = PAD.search(pad_message)
    if not found or (not replace and notes in found[1]):
        return pad_message
    prior = "" if replace or found[1].strip() == "(empty)" else found[1]
    body = "\n\n".join(filter(None, (notes, prior)))[: int(found[3])]
    return (pad_message[: found.start()] + f"<<<SCRATCHPAD\n{body or '(empty)'}\nSCRATCHPAD>>>\n"
            f"{len(body)}/{found[3]} characters" + pad_message[found.end():])


def replay_notes(directory: Path, rows: list[dict]) -> dict[str, str]:
    """Look up the reconstructed opponent notes for each selected decision."""
    groups = defaultdict(dict)
    notes = {}
    for row in rows:
        decision_id = row["id"]
        coordinate = SPOT_ID.fullmatch(decision_id)
        if not coordinate or int(coordinate[2]) != row["card"]["hand"]:
            raise ValueError(f"{decision_id}: expected a replay decision ID matching the prompt's hand number")
        if len(row["messages"]) < 2 or not PAD.search(row["messages"][1]["content"]):
            raise ValueError(f"{decision_id}: expected the private scratchpad message in the replay prompt")
        match_id, hand, seat = coordinate.groups()
        if hand == "1":
            notes[decision_id] = ""
        else:
            groups[match_id][decision_id] = (int(hand), int(seat))
    for match_id, decisions in groups.items():
        path = directory / f"{match_id}.json"
        if not path.is_file():
            raise ValueError(f"missing replay {path}; use the source replay directory for these decisions")
        match = json.loads(path.read_text())
        if match["id"] != match_id:
            raise ValueError(f"{path}: replay ID does not match its filename")
        by_position = replay_betting_evidence(match, decisions.values())
        notes.update((decision_id, by_position[position]) for decision_id, position in decisions.items())
    return notes


def gain(row: dict, answer: str | None = None) -> float:
    if (row["answer"] if answer is None else answer) != "continue" or row.get("equity") is None:
        return 0.0
    c = row["card"]
    return row["equity"] * c["win_value"] + (1 - row["equity"]) * c["lose_value"] - c["fold_value"]


def best_call_answer(row: dict) -> str:
    return "continue" if gain(row, "continue") > 0 else "fold"


def submitted_action(reply: Completion, prompt: str) -> tuple[str, dict]:
    """Require the one legal tool submission used for teacher collection."""
    if reply.truncated or len(reply.tool_calls) != 1 or reply.tool_calls[0]["function"]["name"] != SUBMIT_ACTION:
        return "", {}
    if not isinstance(reply.tool_arguments, (str, dict)):
        return "", {}
    decision, error = parse_tool_call(reply.tool_arguments)
    legal = prompt_legal_actions(prompt)
    if error or legal is None or invalid_action(decision, legal):
        return "", {}
    action = legal.raise_type if decision.type in ("bet", "raise") else decision.type
    return action, {"action": action, "amount": decision.amount}


def summarize(name: str, rows: list[dict]) -> str:
    if not rows:
        return f"  {name:28} none"
    n = len(rows)
    points = sum(map(gain, rows))
    best = sum(r["answer"] == best_call_answer(r) for r in rows)
    agree = sum(r["answer"] == r["card"]["plan"] for r in rows if r["card"]["plan"] != "close")
    clear = sum(r["card"]["plan"] != "close" for r in rows)
    return (f"  {name:28} n {n:4} | {1000 * points / n:+7.1f} rating points per 1000 | best answer {100 * best / n:3.0f}% | "
            f"continues {100 * sum(r['answer'] == 'continue' for r in rows) / n:3.0f}% | agrees with the card "
            f"{100 * agree / max(1, clear):3.0f}% | invalid {sum(r['answer'] == 'invalid' for r in rows)}")


async def run(args: argparse.Namespace) -> None:
    key = os.environ.get("AC2_API_KEY")
    if not key:
        raise SystemExit("export AC2_API_KEY first")
    opening = args.decision == "open"
    make_card = open_card.card if opening else card
    card_hint = open_card.hint if opening else hint
    other = "raise" if opening else "continue"
    spots = [json.loads(line) for line in args.spots.read_text().splitlines() if line.strip()]
    spots = [s for s in spots if s["held_out"] == args.held_out and not s.get("forced") and s.get("share", 1) >= args.min_share]
    rows = [s | {"card": c} for s in spots if (c := make_card(s["messages"])) is not None]
    if opening and (args.hint or args.balance):
        rows = [r for r in rows if r["card"]["clear"]]
    rng = random.Random(0)
    if args.balance:  # as many clear folds as clear continues, nothing close
        folds = [r for r in rows if r["card"]["plan"] == "fold"]
        others = [r for r in rows if r["card"]["plan"] == other]
        size = min(len(folds), len(others))
        rows = rng.sample(folds, size) + rng.sample(others, size)
    rng.shuffle(rows)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit(f"no eligible {args.decision} decisions in {args.spots}")
    params = with_preset({"preset": "event", "seed": 0, "submit_fields": ["action", "amount"]})
    params["submit_fields"] = talk_fields(params["submit_fields"])
    notes = args.notes_file.read_text().strip() if args.notes_file else ""
    try:
        observed = replay_notes(args.opponent_replays, rows) if args.opponent_replays else None
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    gate = asyncio.Semaphore(args.parallel)
    async with httpx.AsyncClient(timeout=httpx.Timeout(900.0, connect=30.0)) as client:
        complete = relay_completion(client, GATEWAY_URL, key, params | {"model_label": args.model})

        async def one(row: dict) -> dict:
            messages = [dict(m) for m in row["messages"]]
            if args.hint:
                messages[0]["content"] = messages[0]["content"] + SEPARATOR + card_hint(row["card"])
            evidence = ""
            if observed is not None:
                evidence = observed[row["id"]]
                scratchpad = "" if row["card"]["hand"] == 1 else "\n\n".join(filter(None, (notes, evidence)))
                messages[1]["content"] = with_notes(messages[1]["content"], scratchpad, replace=True)
            elif notes:
                messages[1]["content"] = with_notes(messages[1]["content"], notes)
            async with gate:
                try:
                    reply = await complete(messages)
                except Exception as exc:  # noqa: BLE001
                    action, parsed, extra = "", {}, {"error": type(exc).__name__}
                else:
                    action, parsed = submitted_action(reply, messages[-1]["content"])
                    extra = {"reasoning": reply.reasoning, "text": reply.text, "out_tokens": reply.out_tokens, "calls": len(reply.tool_calls)}
            if opening:
                answer = open_card.answer(row["card"], parsed)
            else:
                answer = "fold" if action == "fold" else "continue" if action in ("call", "raise", "bet") else "invalid"
            context = {"has_opponent_evidence": bool(evidence)} if observed is not None else {}
            return row | {"answer": answer, "action": action} | context | extra

        done = list(await asyncio.gather(*(one(r) for r in rows)))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("".join(json.dumps({k: v for k, v in r.items() if k != "messages"}) + "\n" for r in done))
    if failures := sum("error" in row for row in done):
        raise SystemExit(f"AC2 requests failed for {failures}/{len(done)} {args.decision} decisions")
    print(f"{args.model}{' with the card' if args.hint else ''}: {len(done)} {args.decision} decisions from "
          f"{'held-out' if args.held_out else 'training'} matches")
    if observed is not None:
        print(f"  prior opponent notes in {sum(r['has_opponent_evidence'] for r in done)}/{len(done)} prompts")
    if opening:
        clear = [r for r in done if r["card"]["clear"]]
        print(f"  small raises {sum(r['answer'] == 'raise' for r in done)}/{len(done)} | "
              f"agrees with clear plans {sum(r['answer'] == r['card']['plan'] for r in clear)}/{len(clear)} | "
              f"invalid {sum(r['answer'] == 'invalid' for r in done)}")
        return
    print(summarize("all", done))
    for name, keep in (
        ("preflop", lambda r: r["card"]["street"] == "preflop"), ("postflop", lambda r: r["card"]["street"] != "preflop"),
        ("40%+ of the stack", lambda r: r["card"]["share"] >= 0.4), ("5-6 players left", lambda r: r["card"]["live"] >= 5),
        ("4 or fewer left", lambda r: r["card"]["live"] <= 4), ("card says fold", lambda r: r["card"]["plan"] == "fold"),
        ("card says continue", lambda r: r["card"]["plan"] == "continue"),
    ):
        print(summarize(name, [r for r in done if keep(r)]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--decision", choices=("call", "open"), default="call")
    parser.add_argument("--spots", type=Path, required=True)
    parser.add_argument("--held-out", action="store_true", help="the held-out matches (default: the training matches)")
    parser.add_argument("--hint", action="store_true", help="append each decision's price card to the system prompt")
    parser.add_argument("--notes-file", type=Path, help="put this text at the top of each decision's scratchpad")
    parser.add_argument("--opponent-replays", type=Path,
                        help="put prior betting counts in the scratchpad from the source replays; hand 1 starts empty")
    parser.add_argument("--balance", action="store_true", help="equal numbers of clear fold and non-fold plans")
    parser.add_argument("--min-share", type=float, default=0.0, help="skip calls smaller than this share of the stack")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--parallel", type=int, default=16)
    parser.add_argument("--out", type=Path)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
