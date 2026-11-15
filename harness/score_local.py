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
import statistics
from pathlib import Path

import httpx

from poker_harness.completions import GATEWAY_URL, cleanup_params, relay_completion, table_rules, with_preset
from poker_harness.engine.scripted import ARCHETYPES, ScriptedPlayer
from poker_harness.game import play_session
from poker_harness.house_bots import HOUSE_STYLES, TALK_MODES, HouseStylePlayer
from poker_harness.opponent_reads import ACTION_GUIDANCE, CLEANUP_GUIDANCE, PLACEMENT_GUIDANCE
from poker_harness.prompts import talk_fields
from poker_harness.seat import ModelSeat
from poker_harness.talk_defense import ACTION_DEFENSE, CLEANUP_DEFENSE


async def one_game(
    client: httpx.AsyncClient,
    key: str,
    model: str,
    seed: int,
    hands: int | None,
    *,
    guidance: str = "",
    talk_mode: str = "normal",
    lineup: str = "house",
) -> dict:
    params = with_preset({"preset": "event", "seed": seed, "submit_fields": ["action", "amount"]})
    if hands:
        params["total_hands"] = hands
    rules = table_rules(params)
    if rules.table_talk:
        params["submit_fields"] = talk_fields(params["submit_fields"])
    rng = random.Random(seed)
    mine = rng.randrange(rules.seats)
    styles = rng.sample(HOUSE_STYLES, rules.seats - 1)
    if lineup == "mixed":
        styles[-2:] = rng.sample(ARCHETYPES, 2)
        rng.shuffle(styles)
    label = params | {"model_label": model}
    players: list = []
    styles_by_seat = {}
    for seat in range(rules.seats):
        name = f"seat{seat}"
        if seat == mine:
            players.append(
                ModelSeat(
                    name,
                    relay_completion(client, GATEWAY_URL, key, label),
                    rules,
                    model=model,
                    cleanup_complete=relay_completion(client, GATEWAY_URL, key, cleanup_params(label)),
                    persona=guidance,
                )
            )
            styles_by_seat[name] = "model"
        else:
            style = styles.pop()
            if style in HOUSE_STYLES:
                players.append(HouseStylePlayer(name, style, seed, talk_mode=talk_mode))
                styles_by_seat[name] = style.removeprefix("house_")
            else:
                players.append(ScriptedPlayer(name, style, seed))
                styles_by_seat[name] = style
    log = await play_session(f"score-{seed}", seed, players, rules)
    log["scored_seat"] = f"seat{mine}"
    log["lineup"] = styles_by_seat
    return log


def summarize(log: dict) -> dict:
    seat = log["scored_seat"]
    mine = [a for hand in log["hands"] for a in hand["actions"] if a["player"] == seat]
    cleanup = [
        hand["cleanup"] for hand in log.get("private", {}).get(seat, {}).get("hands", {}).values() if "cleanup" in hand
    ]
    return {
        "seed": log["session_info"]["seed"],
        "seat": seat,
        "place": log["ranking"]["places"][seat],
        "chips": log["final_stacks"][seat],
        "busted_on": log["eliminated_on_hand"].get(seat),
        "hands": len(log["hands"]),
        "decisions": len(mine),
        "failed": sum(bool(a["model_response"].get("format_error")) for a in mine),
        "cleanup_turns": len(cleanup),
        "cleanup_updated": sum(turn["result"] == "updated" for turn in cleanup),
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
    arms = {"official": ""}
    if args.opponent_read_ablation:
        arms = {
            "control": PLACEMENT_GUIDANCE,
            "opponent_reads": "\n\n".join((PLACEMENT_GUIDANCE, ACTION_GUIDANCE, CLEANUP_GUIDANCE)),
        }
    elif args.table_talk_defense_ablation:
        arms = {"control": "", "table_talk_defense": f"{ACTION_DEFENSE}\n\n{CLEANUP_DEFENSE}"}
    folders = {arm: args.out / arm if len(arms) > 1 else args.out for arm in arms}
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=30.0)) as client:

        async def guarded(seed: int, arm: str, guidance: str) -> dict:
            async with gate:
                log = await one_game(
                    client, key, args.model, seed, args.hands, guidance=guidance, talk_mode=args.talk_mode,
                    lineup=args.lineup,
                )
            row = summarize(log) | {"arm": arm, "talk_mode": args.talk_mode}
            log["experiment"] = {"arm": arm, "talk_mode": args.talk_mode, "deployment": args.model}
            (folders[arm] / f"{log['game_id']}.json").write_text(json.dumps(log, indent=1))
            print(json.dumps(row), flush=True)
            return row

        return list(await asyncio.gather(*(guarded(seed, arm, hint) for seed in seeds for arm, hint in arms.items())))


def compare_arms(rows: list[dict], treatment_arm: str = "opponent_reads") -> dict:
    control = {row["seed"]: row for row in rows if row["arm"] == "control"}
    treatment = {row["seed"]: row for row in rows if row["arm"] == treatment_arm}
    shared = sorted(control.keys() & treatment.keys())
    changes = [control[seed]["place"] - treatment[seed]["place"] for seed in shared]
    return {
        "paired_games": len(changes),
        "unpaired_games": len(control.keys() ^ treatment.keys()),
        "mean_places_gained": round(statistics.fmean(changes), 3) if changes else None,
        "improved": sum(change > 0 for change in changes),
        "tied": sum(change == 0 for change in changes),
        "worsened": sum(change < 0 for change in changes),
    }


def print_summary(label: str, rows: list[dict]) -> None:
    n = len(rows)
    decisions = sum(r["decisions"] for r in rows)
    print(
        f"\n{label}: {n} games | mean place {sum(r['place'] for r in rows) / n:.2f} of 6 | "
        f"busted {sum(r['busted_on'] is not None for r in rows)}/{n} | places {sorted(r['place'] for r in rows)} | "
        f"mean chips {sum(r['chips'] for r in rows) / n:.0f} | failed decisions {sum(r['failed'] for r in rows)}/{decisions} | "
        f"cleanup updated {sum(r['cleanup_updated'] for r in rows)}/{sum(r['cleanup_turns'] for r in rows)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="your deployment, <project>/<name>")
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--parallel", type=int, default=8, help="games played at once")
    parser.add_argument("--seed", type=int, default=20261114, help="first seed; the same seeds give the same deals")
    parser.add_argument("--hands", type=int, help="hands per game (default: the event's 30)")
    parser.add_argument("--out", type=Path, default=Path("runs/score"))
    ablations = parser.add_mutually_exclusive_group()
    ablations.add_argument("--opponent-read-ablation", action="store_true",
                           help="run two arms per seed with the same placement guidance, adding seat reads to one arm")
    ablations.add_argument("--table-talk-defense-ablation", action="store_true",
                           help="compare the official prompt with extra instructions to ignore chatter")
    parser.add_argument("--talk-mode", choices=TALK_MODES, default="normal", help="scripted opponents' table talk")
    parser.add_argument("--lineup", choices=("house", "mixed"), default="house",
                        help="mixed replaces two house imitations with generic scripted opponents")
    args = parser.parse_args()
    if args.games < 1 or args.parallel < 1 or (args.hands is not None and args.hands < 1):
        parser.error("--games, --parallel and --hands must be positive")
    if args.opponent_read_ablation and args.hands == 1:
        parser.error("the opponent-read ablation needs at least two hands")
    rows = asyncio.run(run(args))
    for arm in dict.fromkeys(row["arm"] for row in rows):
        print_summary(f"{args.model} ({arm})", [row for row in rows if row["arm"] == arm])
    if args.opponent_read_ablation:
        print(json.dumps(compare_arms(rows)))
    elif args.table_talk_defense_ablation:
        print(json.dumps(compare_arms(rows, "table_talk_defense")))


if __name__ == "__main__":
    main()
