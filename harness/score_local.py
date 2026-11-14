"""Score a deployment the way the arena does: full 30-hand event games against the six house-bot imitations.

    AC2_API_KEY=<key> uv run python score_local.py --model vals-hackathon/entry --games 8

Each game seats your deployment in a different seat against five of the six scripted imitations in
`poker_harness/house_bots.py` (no opponent model calls) and plays the official event rules. It reports the finishing
place, bust rate and failed decisions, which is what the arena's rating is built from. The imitations are
approximations of the real house bots, so use this to compare your own checkpoints with each other, not to predict
an arena rating.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
from pathlib import Path

import httpx

from poker_harness.completions import GATEWAY_URL, cleanup_params, relay_completion, table_rules, with_preset
from poker_harness.game import play_session
from poker_harness.house_bots import HOUSE_STYLES, HouseStylePlayer
from poker_harness.prompts import talk_fields
from poker_harness.seat import ModelSeat


async def one_game(client: httpx.AsyncClient, key: str, model: str, seed: int, hands: int | None) -> dict:
    params = with_preset({"preset": "event", "seed": seed, "submit_fields": ["action", "amount"]})
    if hands:
        params["total_hands"] = hands
    rules = table_rules(params)
    if rules.table_talk:
        params["submit_fields"] = talk_fields(params["submit_fields"])
    rng = random.Random(seed)
    mine = rng.randrange(rules.seats)
    styles = rng.sample(HOUSE_STYLES, rules.seats - 1)
    label = params | {"model_label": model}
    players: list = []
    for seat in range(rules.seats):
        if seat == mine:
            players.append(
                ModelSeat(
                    f"seat{seat}",
                    relay_completion(client, GATEWAY_URL, key, label),
                    rules,
                    model=model,
                    cleanup_complete=relay_completion(client, GATEWAY_URL, key, cleanup_params(label)),
                )
            )
        else:
            players.append(HouseStylePlayer(f"seat{seat}", styles.pop(), seed))
    log = await play_session(f"score-{seed}", seed, players, rules)
    log["scored_seat"] = f"seat{mine}"
    log["lineup"] = {p.name: getattr(p, "style", "model") for p in players}
    return log


def summarize(log: dict) -> dict:
    seat = log["scored_seat"]
    mine = [a for hand in log["hands"] for a in hand["actions"] if a["player"] == seat]
    return {
        "seed": log["session_info"]["seed"],
        "seat": seat,
        "place": log["ranking"]["places"][seat],
        "chips": log["final_stacks"][seat],
        "busted_on": log["eliminated_on_hand"].get(seat),
        "hands": len(log["hands"]),
        "decisions": len(mine),
        "failed": sum(bool(a["model_response"].get("format_error")) for a in mine),
        "winner": min(log["ranking"]["places"], key=log["ranking"]["places"].get),
        "winner_style": log["lineup"][min(log["ranking"]["places"], key=log["ranking"]["places"].get)],
    }


async def run(args: argparse.Namespace) -> list[dict]:
    key = os.environ.get("AC2_API_KEY")
    if not key:
        raise SystemExit("export AC2_API_KEY first")
    seeds = [args.seed + i for i in range(args.games)]
    args.out.mkdir(parents=True, exist_ok=True)
    gate = asyncio.Semaphore(args.parallel)
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=30.0)) as client:

        async def guarded(seed: int) -> dict:
            async with gate:
                log = await one_game(client, key, args.model, seed, args.hands)
            (args.out / f"{log['game_id']}.json").write_text(json.dumps(log, indent=1))
            row = summarize(log)
            print(json.dumps(row), flush=True)
            return row

        return list(await asyncio.gather(*(guarded(s) for s in seeds)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="your deployment, <project>/<name>")
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--parallel", type=int, default=8, help="games played at once")
    parser.add_argument("--seed", type=int, default=20261114, help="first seed; the same seeds give the same deals")
    parser.add_argument("--hands", type=int, help="hands per game (default: the event's 30)")
    parser.add_argument("--out", type=Path, default=Path("runs/score"))
    args = parser.parse_args()
    rows = asyncio.run(run(args))
    n = len(rows)
    decisions = sum(r["decisions"] for r in rows)
    print(
        f"\n{args.model}: {n} games | mean place {sum(r['place'] for r in rows) / n:.2f} of 6 | "
        f"busted {sum(r['busted_on'] is not None for r in rows)}/{n} | places {sorted(r['place'] for r in rows)} | "
        f"mean chips {sum(r['chips'] for r in rows) / n:.0f} | failed decisions {sum(r['failed'] for r in rows)}/{decisions}"
    )


if __name__ == "__main__":
    main()
