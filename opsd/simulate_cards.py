"""Compare opening and call cards over paired, full-length CPU games with a shared scripted policy."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import open_card
import price_card
import terminal_calls as terminal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness" / "src"))
from poker_harness.completions import EVENT_PARAMS, table_rules  # noqa: E402
from poker_harness.engine.scripted import ARCHETYPES, ScriptedPlayer  # noqa: E402
from poker_harness.engine.table import Decision, DecisionContext  # noqa: E402
from poker_harness.game import play_session  # noqa: E402
from poker_harness.house_bots import HOUSE_STYLES, HouseStylePlayer  # noqa: E402
from poker_harness.placement import rating_points  # noqa: E402
from poker_harness.prompts import decision_prompt  # noqa: E402

RULES = table_rules(EVENT_PARAMS)
ARMS = ("scripted", "chip_price", "placement_price", "open_only", "open_and_placement")
LINEUPS = ("house", "mixed", "tight")
RANGE_ARMS = ("range_field_only", "range_pluribus", "open_and_range")


@lru_cache(maxsize=128)
def terminal_worlds(spot: terminal.Call) -> list[tuple[str, int]]:
    return terminal.worlds(spot, 256, repr(spot))


class CardPlayer(HouseStylePlayer):
    def __init__(
        self, name: str, seed: int, arm: str, book: dict | None = None
    ) -> None:
        super().__init__(name, "house_pete", seed, talk_mode="muted")
        self.arm = arm
        self.changed = 0
        self.first_in = 0
        self.opened = 0
        self.book = book
        self.starting: dict[int, int] = {}
        self.range_decisions = 0

    def begin_hand(self, info: dict) -> None:
        self.starting = {
            int(name.removeprefix("seat")): stack
            for name, stack in info["stacks"].items()
        }

    def range_action(
        self, ctx: DecisionContext, messages: list[dict], book: dict
    ) -> str | None:
        bets = [
            a
            for a in ctx.hand_actions
            if a["street"] == ctx.street and a["action"]["type"] in ("bet", "raise")
        ]
        if not bets:
            return None
        spot = terminal.public_call(
            messages, self.starting, bets[-1]["seat"], len(bets) > 1
        )
        if spot is None:
            return None
        if (
            ctx.street == "river"
            and not spot.shove
            and ctx.legal.raise_type
            and terminal.hand_bucket(spot.hole, spot.board)
            in {
                "monster",
                "trips",
                "two_pair",
            }
        ):
            return None
        _, gains = terminal.values(spot, price_card.tables()["value"])
        edges = terminal.estimate(
            spot,
            terminal_worlds(spot),
            book,
            gains,
            use_pluribus=self.arm != "range_field_only",
        )
        return terminal.robust_action(edges)

    async def decide(self, ctx: DecisionContext, history: str) -> Decision:
        default = await super().decide(ctx, history)
        legal = ctx.legal
        if not legal.can_fold:
            return default
        first_in = (
            ctx.street == "preflop"
            and legal.raise_type == "raise"
            and all(action["action"]["type"] == "fold" for action in ctx.hand_actions)
        )
        decision = default
        if first_in:
            self.first_in += 1
            if self.arm in ("open_only", "open_and_placement", "open_and_range"):
                c = open_card.card([{"role": "user", "content": decision_prompt(ctx)}])
                if c and c["clear"]:
                    amount = c["raise_to"] if c["plan"] == "raise" else None
                    decision = Decision(c["plan"], amount, default.model_response)
            self.opened += decision.type == "raise"
        elif self.arm in (
            "chip_price",
            "placement_price",
            "open_and_placement",
            *RANGE_ARMS,
        ):
            stack = next(s.stack for s in ctx.seats if s.seat == ctx.seat)
            if legal.call_amount >= 0.12 * stack or ctx.street == "river":
                messages = [{"role": "user", "content": decision_prompt(ctx)}]
                c = price_card.card(messages)
                if c:
                    threshold = c["price"] if self.arm == "chip_price" else c["need"]
                    edge = c["win"] - threshold
                    if edge >= price_card.CLOSE and default.type == "fold":
                        decision = Decision(
                            "call", model_response=default.model_response
                        )
                    elif edge <= -price_card.CLOSE and default.type != "fold":
                        decision = Decision(
                            "fold", model_response=default.model_response
                        )
                if (
                    self.book
                    and self.arm in RANGE_ARMS
                    and default.type in ("fold", "call")
                    and (action := self.range_action(ctx, messages, self.book))
                ):
                    self.range_decisions += 1
                    decision = Decision(action, model_response=default.model_response)
        self.changed += (decision.type, decision.amount) != (
            default.type,
            default.amount,
        )
        return decision


async def play(seed: int, lineup: str, arm: str, book: dict | None = None) -> dict:
    rng = random.Random(seed)
    hero = rng.randrange(RULES.seats)
    styles = [style for style in HOUSE_STYLES if style != "house_pete"]
    if lineup == "mixed":
        styles[-2:] = rng.sample(ARCHETYPES, 2)
    elif lineup == "tight":
        styles = ["nit", "nit", "house_alice", "house_bob", "house_tim"]
    rng.shuffle(styles)
    policy = CardPlayer(f"seat{hero}", seed, arm, book)
    players = []
    for seat in range(RULES.seats):
        name = f"seat{seat}"
        if seat == hero:
            players.append(policy)
        else:
            style = styles.pop()
            opponent = (
                HouseStylePlayer(name, style, seed, talk_mode="muted")
                if style in HOUSE_STYLES
                else ScriptedPlayer(name, style, seed)
            )
            players.append(opponent)
    log = await play_session(f"cards-{lineup}-{seed}", seed, players, RULES)
    place = log["ranking"]["places"][policy.name]
    return {
        "seed": seed,
        "lineup": lineup,
        "arm": arm,
        "place": place,
        "points": rating_points(place, RULES.seats),
        "bust": policy.name in log["eliminated_on_hand"],
        "changed_actions": policy.changed,
        "first_in_opportunities": policy.first_in,
        "opens": policy.opened,
        "range_decisions": policy.range_decisions,
    }


def paired(job: tuple[int, str, dict | None]) -> list[dict]:
    seed, lineup, book = job
    arms = (
        ("scripted", "placement_price", "open_and_placement", *RANGE_ARMS)
        if book
        else ARMS
    )
    return [asyncio.run(play(seed, lineup, arm, book)) for arm in arms]


def summarize(pairs: list[list[dict]], control: str) -> dict:
    metrics = {}
    arms = tuple(row["arm"] for row in pairs[0])
    baseline = arms.index(control)
    for index, arm in enumerate(arms):
        rows = [pair[index] for pair in pairs]
        deltas = [pair[index]["points"] - pair[baseline]["points"] for pair in pairs]
        mean = statistics.fmean(deltas)
        n = len(rows)
        margin = 1.96 * statistics.stdev(deltas) / n**0.5 if n > 1 else None
        metrics[arm] = {
            "games": n,
            "points_per_game": round(
                statistics.fmean(row["points"] for row in rows), 4
            ),
            "busts": sum(row["bust"] for row in rows),
            "places": dict(sorted(Counter(r["place"] for r in rows).items())),
            "changed_actions": sum(row["changed_actions"] for row in rows),
            "first_in_opportunities": sum(
                row["first_in_opportunities"] for row in rows
            ),
            "opens": sum(row["opens"] for row in rows),
            "range_decisions": sum(row.get("range_decisions", 0) for row in rows),
            f"delta_over_{control}": round(mean, 4),
            "paired_95_interval": None
            if margin is None
            else [round(mean - margin, 4), round(mean + margin, 4)],
        }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--games", type=int, default=256, help="paired seeds per lineup"
    )
    parser.add_argument("--seed", type=int, default=20272000)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--out", type=Path, default=Path("harness/runs/card-sims.jsonl")
    )
    parser.add_argument("--lineups", choices=LINEUPS, nargs="+", default=LINEUPS)
    parser.add_argument(
        "--terminal-ranges",
        type=Path,
        help="Compare terminal teachers using a trained ranges.json",
    )
    args = parser.parse_args()
    if min(args.games, args.workers) < 1:
        parser.error("--games and --workers must be positive")
    book = (
        json.loads(args.terminal_ranges.read_text()) if args.terminal_ranges else None
    )
    jobs = [
        (args.seed + index * args.games + offset, lineup, book)
        for index, lineup in enumerate(dict.fromkeys(args.lineups))
        for offset in range(args.games)
    ]
    pairs = []
    with ProcessPoolExecutor(args.workers) as pool:
        for pair in pool.map(paired, jobs, chunksize=4):
            pairs.append(pair)
            if len(pairs) % 128 == 0:
                print(f"{len(pairs)}/{len(jobs)} paired seeds finished", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(row) + "\n" for pair in pairs for row in pair)
    )
    groups = {"all": pairs} | {
        lineup: [pair for pair in pairs if pair[0]["lineup"] == lineup]
        for lineup in args.lineups
    }
    summary = {
        "seed": args.seed,
        "groups": {
            group: {
                "policies": summarize(part, "placement_price"),
                "scripted_comparison": summarize(part, "scripted"),
            }
            for group, part in groups.items()
        },
    }
    args.out.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    for group, result in summary["groups"].items():
        for arm, row in result["policies"].items():
            print(
                f"{group:5} {arm:18} {row['points_per_game']:+.4f} points/game | {row['busts']}/{row['games']} busts "
                f"| vs placement price {row['delta_over_placement_price']:+.4f} {row['paired_95_interval']}"
            )


if __name__ == "__main__":
    main()
