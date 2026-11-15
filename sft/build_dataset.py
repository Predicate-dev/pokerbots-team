"""Build a supervised (SFT) dataset from decisions rebuilt by `harness/clone_from_replays.py`.

    (cd harness && uv run python clone_from_replays.py --replays <dir> --entry house-professor-pete --out runs/clone_pete.jsonl)
    uv run python sft/build_dataset.py --examples harness/runs/clone_pete.jsonl --dataset pokerbots-sft-pete            # dry run
    uv run python sft/build_dataset.py --examples harness/runs/clone_pete.jsonl --dataset pokerbots-sft-pete --upload

Each task's `input` is one full conversation: the official event prompt for a spot, then the action the cloned
entry took there as a single `submit_action` call, after one factual line of thinking (replays publish no
reasoning, so the line only restates the spot and the move; see `brief_reason`).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from ac2.runtime import Item, Task, Tool
from ac2.sdk import Client
from pydantic import TypeAdapter

ITEMS = TypeAdapter(list[Item])
TOOLS = [Tool.model_validate(t).model_dump() for t in json.loads((Path(__file__).parents[1] / "opsd" / "tools.json").read_text())]
BETWEEN = "\n\n"  # what the base model writes between its thinking and a tool call


def items(messages: list[dict]) -> list[dict]:
    """Chat messages in the item form AC2 stores conversations in."""
    out: list[dict] = []
    for m in messages:
        if m["role"] in ("system", "user"):
            out.append({"type": "message", "role": m["role"], "content": m["content"]})
        elif m["role"] == "assistant":
            out.append({"type": "message", "role": "assistant", "content": m.get("content") or BETWEEN})
            for call in m.get("tool_calls") or []:
                out.append({"type": "function_call", "call_id": call["id"], "name": call["function"]["name"],
                            "arguments": call["function"]["arguments"]})
        elif m["role"] == "tool":
            out.append({"type": "function_call_output", "call_id": m["tool_call_id"], "output": m["content"]})
        else:
            raise ValueError(f"unexpected role {m['role']!r}")
    return out


APPLIED = "Applied: {move}."


def _move(example: dict) -> str:
    action = example["action"]
    return action["action"] + (f" to {action['amount']}" if "amount" in action else "")


def brief_reason(example: dict) -> str:
    """One factual line of thinking for the target. Qwen is served with thinking on, and a target with no thinking
    at all teaches it to write the tool call inside the thinking block, where the server never parses it (measured
    on the first run: every reply had no tool call). A short line keeps the block and then closes it."""
    action = example["action"]
    move = action["action"] + (f" to {action['amount']}" if "amount" in action else "")
    # The cards are left out on purpose: a first version named them here, and the model then wrote lines like
    # "raise with qh js preflop." into its table talk, telling the table its hand.
    return f"{example['street'].capitalize()}: {move}."


RANKS = "23456789TJQKA"
STRONG_CLAIMS = ("raising with pocket aces.", "raising with pocket kings.", "raise with ak suited.")
WEAK_CLAIMS = ("just stealing with seven deuce offsuit.", "stealing the blinds with junk.")


def premium(hole: list[str]) -> bool:
    """AA, KK, QQ or AK."""
    ranks = sorted((c[0] for c in hole), key=RANKS.index)
    return ranks in (["A", "A"], ["K", "K"], ["Q", "Q"], ["K", "A"])


def talk_line(example: dict, rng: random.Random) -> str:
    """The table-talk recipe, applied to preflop raises only. Measured on base Qwen (harness/talk_test.py): a
    specific strong-hand claim raised its fold rate against a raise from 40% to 72%, and a weak-hand claim raised
    its calls of an all-in from 28% to 46%. So: half the time, a raise without a premium hand claims one, and a
    raise with a premium hand claims junk. Everything else stays silent, and the model's real cards are never named."""
    if example["street"] != "preflop" or example["action"]["action"] not in ("bet", "raise") or rng.random() >= 0.5:
        return ""
    return rng.choice(WEAK_CLAIMS if premium(example["hole"]) else STRONG_CLAIMS)


def cleanup_task(example: dict, index: int) -> Task:
    """An after-hand turn: the conversation up to the cleanup prompt, then one `update_scratchpad` call that writes
    the seat ledger with this hand added (`harness/clone_from_replays.py --ledger`)."""
    conversation = items(example["messages"]) + [
        {"type": "reasoning", "content": f"Hand {example['hand']} is over. Add its actions to each seat's counters, "
                                         "update the stacks and reads, and keep the goal line."},
        {"type": "message", "role": "assistant", "content": BETWEEN},
        {"type": "function_call", "call_id": f"call-target-{index}", "name": "update_scratchpad",
         "arguments": json.dumps({"contents": example["contents"]})},
        {"type": "function_call_output", "call_id": f"call-target-{index}", "output": "Scratchpad updated."},
    ]
    return Task(
        input=ITEMS.validate_json(json.dumps(conversation)),
        env_params={"tools": TOOLS, "source": f"{example['match']}:{example['hand']}:cleanup"},
    )


def task(example: dict, index: int, say: str = "") -> Task:
    if example.get("kind") == "cleanup":
        return cleanup_task(example, index)
    arguments = example["action"] | {"say": say} if say else example["action"]
    conversation = items(example["messages"]) + [
        {"type": "reasoning", "content": brief_reason(example)},
        {"type": "message", "role": "assistant", "content": BETWEEN},
        {"type": "function_call", "call_id": f"call-target-{index}", "name": "submit_action",
         "arguments": json.dumps(arguments)},
        # The tool's reply closes the assistant turn. Without anything after the call, the first models kept writing
        # past it (repeated calls, stray tags) and 10% of their decisions were unreadable.
        {"type": "function_call_output", "call_id": f"call-target-{index}", "output": APPLIED.format(move=_move(example))},
    ]
    return Task(
        input=ITEMS.validate_json(json.dumps(conversation)),
        env_params={"tools": TOOLS, "source": f"{example['match']}:{example['hand']}:{example['street']}"},
    )


def build(examples: list[dict], max_fold_share: float, seed: int = 0, talk: bool = False) -> list[Task]:
    """All non-fold decisions, and folds capped at `max_fold_share` of the result so they do not swamp it."""
    rng = random.Random(seed)
    cleanup = [e for e in examples if e.get("kind") == "cleanup"]
    examples = [e for e in examples if e.get("kind") != "cleanup"]
    folds = [e for e in examples if e["action"]["action"] == "fold"]
    rest = [e for e in examples if e["action"]["action"] != "fold"] + cleanup
    if 0 <= max_fold_share < 1:
        folds = rng.sample(folds, min(len(folds), int(len(rest) * max_fold_share / (1 - max_fold_share))))
    chosen = folds + rest
    rng.shuffle(chosen)
    return [
        task(e, i, talk_line(e, rng) if talk and e.get("kind") != "cleanup" else "") for i, e in enumerate(chosen)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default="vals-hackathon")
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--max-fold-share", type=float, default=1.0, help="cap on the share of folds (default: keep all)")
    parser.add_argument("--talk", action="store_true", help="add the table-talk recipe to preflop raises (see talk_line)")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    examples = [json.loads(line) for line in args.examples.read_text().splitlines() if line.strip()]
    tasks = build(examples, args.max_fold_share, talk=args.talk)
    said = sum("say" in json.loads(t.input[-2].arguments) for t in tasks)
    actions = Counter(json.loads(t.input[-2].arguments).get("action", "update_scratchpad") for t in tasks)
    print(json.dumps({"examples": len(examples), "tasks": len(tasks), "actions": dict(actions), "with_talk": said}))
    if not args.upload:
        print("dry run: nothing uploaded (pass --upload to create the dataset)")
        return
    client = Client(project=args.project)
    client.datasets.create(args.dataset, exist_ok=True)
    for start in range(0, len(tasks), 500):
        client.datasets.add_tasks(args.dataset, tasks=tasks[start : start + 500])
    print(f"uploaded {len(tasks)} tasks to {args.dataset}")


if __name__ == "__main__":
    main()
