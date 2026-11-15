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

from poker_harness.arena_lineups import FIELD, ArenaRoster, recent_rosters
from poker_harness.balanced_play import BALANCED_PLAY
from poker_harness.completions import GATEWAY_URL, cleanup_params, relay_completion, table_rules, with_preset
from poker_harness.engine.scripted import ARCHETYPES, ScriptedPlayer
from poker_harness.game import play_session
from poker_harness.house_bots import HOUSE_STYLES, TALK_MODES, HouseStylePlayer
from poker_harness.opponent_reads import ACTION_GUIDANCE, CLEANUP_GUIDANCE, PLACEMENT_GUIDANCE
from poker_harness.placement import rating_points
from poker_harness.preflop import PreflopAction, opportunities, record_folds
from poker_harness.preflop_reference import reference_hint
from poker_harness.prompts import talk_fields
from poker_harness.seat import Complete, Completion, Message, ModelSeat
from poker_harness.talk_defense import ACTION_DEFENSE, CLEANUP_DEFENSE, SPECIFIC_CLAIMS


def with_preflop_reference(complete: Complete) -> Complete:
    async def guided(messages: list[Message]) -> str | Completion:
        system, user = messages[0]["content"], messages[-1]["content"]
        if reference := await asyncio.to_thread(reference_hint, system, user):
            messages = [messages[0] | {"content": f"{system}\n\n{reference}"}, *messages[1:]]
        return await complete(messages)

    return guided


# Reproducible approximation for older comparisons. Use --arena-replays for the current house composition.
ARENA_FIELD_SEATS = ((0, 5), (1, 3), (3, 5), (4, 10), (5, 2))


async def _no_cleanup(messages: list) -> Completion:
    """Field seats skip the after-hand scratchpad turn: it would double the model calls for seats we don't score."""
    return Completion(text="")


class KeptNotesSeat(ModelSeat):
    """A model seat whose scratchpad always carries a given block after its first hand, whatever the model wrote.

    This previews a notes block before training a model to keep it: the model plays under the official prompt and
    writes its own notes, and after every after-hand update the block is put back under its checklist if missing."""

    kept = ""

    async def cleanup(self, summary: dict) -> dict:
        result = await super().cleanup(summary)
        pad = self.scratchpad
        if self.kept and self.kept not in pad:
            head, _, tail = pad.partition("\n\n")
            pad = f"{head}\n\n{self.kept}\n\n{tail}" if head.startswith("CALL CHECKLIST") else f"{self.kept}\n\n{pad}"
            self.scratchpad = pad.strip()[: self.rules.scratchpad_chars]
        return result


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
    field_model: str = "",
    field_thinking: bool = True,
    keep_notes: str = "",
    arena_rosters: tuple[ArenaRoster, ...] = (),
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
    roster = None
    if lineup == "mixed":
        styles[-2:] = rng.sample(ARCHETYPES, 2)
        rng.shuffle(styles)
    if lineup == "arena":  # some seats are other teams' bots: `field_model` under the official prompt
        if arena_rosters:
            roster = random.Random(seed).choice(arena_rosters)
            mine = roster.hero
        else:
            field_seats = rng.choices(*zip(*ARENA_FIELD_SEATS, strict=True))[0]
            styles[:field_seats] = [FIELD] * field_seats
            rng.shuffle(styles)
    label = params | {"model_label": model}
    players: list = []
    styles_by_seat = {}
    field_by_seat = {}
    house_by_seat = {}
    for seat in range(rules.seats):
        name = f"seat{seat}"
        if seat == mine:
            complete = relay_completion(client, GATEWAY_URL, key, label)
            if guidance == BALANCED_PLAY:
                complete = with_preflop_reference(complete)
            scored = (KeptNotesSeat if keep_notes else ModelSeat)(
                name,
                complete,
                rules,
                model=model,
                cleanup_complete=relay_completion(client, GATEWAY_URL, key, cleanup_params(label)),
                persona=guidance,
            )
            if keep_notes:
                scored.kept = keep_notes
            players.append(scored)
            styles_by_seat[name] = "model"
        else:
            style = roster.seats[seat] if roster else styles.pop()
            if roster and style != FIELD:
                house_by_seat[name] = style
            if style == FIELD:
                # Several deployments can share the field seats (comma-separated), so opponents don't queue on one GPU.
                fields = [m.strip() for m in field_model.split(",") if m.strip()]
                theirs_model = fields[(seed + seat) % len(fields)]
                field_by_seat[name] = theirs_model
                theirs = params | {"model_label": theirs_model}
                if not field_thinking:  # about ten times fewer tokens per opponent decision, so games finish sooner
                    theirs = {k: v for k, v in theirs.items() if k != "chat_template_kwargs"} | {"thinking": False}
                players.append(
                    ModelSeat(name, relay_completion(client, GATEWAY_URL, key, theirs), rules, model=theirs_model,
                              cleanup_complete=_no_cleanup)
                )
                styles_by_seat[name] = "field"
            elif style in HOUSE_STYLES:
                players.append(HouseStylePlayer(name, style, seed, talk_mode=talk_mode))
                styles_by_seat[name] = style.removeprefix("house_")
                house_by_seat[name] = style
            else:
                players.append(ScriptedPlayer(name, style, seed))
                styles_by_seat[name] = style
    log = await play_session(f"score-{seed}", seed, players, rules)
    log["scored_seat"] = f"seat{mine}"
    log["lineup"] = styles_by_seat
    log["house_opponents"] = house_by_seat
    log["arena_roster"] = roster.match_id if roster else None
    log["matchup"] = {
        "scored_seat": log["scored_seat"], "lineup": styles_by_seat, "field_models": field_by_seat,
        "field_thinking": field_thinking if field_by_seat else None, "talk_mode": talk_mode,
        "hands": rules.total_hands,
    }
    return log


def summarize(log: dict) -> dict:
    seat = log["scored_seat"]
    mine = [a for hand in log["hands"] for a in hand["actions"] if a["player"] == seat]
    preflop_spots = {}
    for hand in log["hands"]:
        preflop = [a for a in hand["actions"] if a["street"] == "preflop"]
        dealt = [player["seat"] for player in hand["starting_state"].values()]
        moves = (
            PreflopAction(a["seat"], a["action"]["type"], a["call_amount"], a["action"].get("amount"))
            for a in preflop
        )
        spots = opportunities(moves, dealt, hand["button_seat"], log["session_info"]["big_blind"])
        for action, tags in zip(preflop, spots, strict=True):
            response = action["model_response"]
            if action["player"] == seat and not (response.get("format_error") or response.get("fallback_used")):
                record_folds(preflop_spots, tags, action["action"]["type"])
    cleanup = [
        hand["cleanup"] for hand in log.get("private", {}).get(seat, {}).get("hands", {}).values() if "cleanup" in hand
    ]
    return {
        "seed": log["session_info"]["seed"],
        "seat": seat,
        "place": log["ranking"]["places"][seat],
        "rating_points": rating_points(log["ranking"]["places"][seat], len(log["ranking"]["places"])),
        "chips": log["final_stacks"][seat],
        "busted_on": log["eliminated_on_hand"].get(seat),
        "hands": len(log["hands"]),
        "decisions": len(mine),
        "failed": sum(bool(a["model_response"].get("format_error")) for a in mine),
        "fallbacks": sum(bool(a["model_response"].get("fallback_used")) for a in mine),
        "preflop_spots": preflop_spots,
        "cleanup_turns": len(cleanup),
        "cleanup_updated": sum(turn["result"] == "updated" for turn in cleanup),
        "winner": min(log["ranking"]["places"], key=log["ranking"]["places"].get),
        "winner_style": log["lineup"][min(log["ranking"]["places"], key=log["ranking"]["places"].get)],
        "house_count": len(log.get("house_opponents", {})),
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
            "opponent_reads": f"{PLACEMENT_GUIDANCE}\n\n{ACTION_GUIDANCE}\n\n{CLEANUP_GUIDANCE}",
        }
    elif args.table_talk_defense_ablation:
        arms = {"control": "", "table_talk_defense": f"{ACTION_DEFENSE}\n\n{CLEANUP_DEFENSE}"}
        if args.talk_mode == "false-claims":
            arms["specific_claims"] = SPECIFIC_CLAIMS
    elif args.balanced_play_ablation:
        arms = {"control": "", "balanced_play": BALANCED_PLAY}
    elif args.compare_model:
        arms = {"control": "", "candidate": ""}
    elif args.guidance_file:  # preview a strategy hint: the text is appended to the model's system prompt
        arms = {"hinted": args.guidance_file.read_text().strip()}
    folders = {arm: args.out / arm if len(arms) > 1 else args.out for arm in arms}
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    keep_notes = args.keep_notes.read_text().strip() if args.keep_notes else ""
    rosters = recent_rosters(args.arena_replays, args.arena_entry) if args.arena_replays else ()
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=30.0)) as client:

        async def guarded(seed: int, arm: str, guidance: str) -> dict:
            model = args.compare_model if args.compare_model and arm == "control" else args.model
            async with gate:
                log = await one_game(
                    client, key, model, seed, args.hands, guidance=guidance, talk_mode=args.talk_mode,
                    lineup=args.lineup, field_model=args.field_model, field_thinking=not args.field_no_thinking,
                    keep_notes=keep_notes if arm != "control" else "",
                    arena_rosters=rosters,
                )
            row = summarize(log) | {"arm": arm, "talk_mode": args.talk_mode, "deployment": model}
            log["experiment"] = {"arm": arm, "talk_mode": args.talk_mode, "deployment": model}
            (folders[arm] / f"{log['game_id']}.json").write_text(json.dumps(log, indent=1))
            print(json.dumps(row), flush=True)
            return row

        return list(await asyncio.gather(*(guarded(seed, arm, hint) for seed in seeds for arm, hint in arms.items())))


def compare_arms(rows: list[dict], treatment_arm: str = "opponent_reads", *, control_arm: str = "control") -> dict:
    control = {row["seed"]: row for row in rows if row["arm"] == control_arm}
    treatment = {row["seed"]: row for row in rows if row["arm"] == treatment_arm}
    shared = sorted(control.keys() & treatment.keys())
    changes = [control[seed]["place"] - treatment[seed]["place"] for seed in shared]
    points = [treatment[seed]["rating_points"] - control[seed]["rating_points"] for seed in shared]
    return {
        "paired_games": len(changes),
        "unpaired_games": len(control.keys() ^ treatment.keys()),
        "mean_places_gained": round(statistics.fmean(changes), 3) if changes else None,
        "rating_points_gained_per_game": round(statistics.fmean(points), 3) if points else None,
        "improved": sum(change > 0 for change in changes),
        "tied": sum(change == 0 for change in changes),
        "worsened": sum(change < 0 for change in changes),
    }


def print_summary(label: str, rows: list[dict]) -> None:
    n = len(rows)
    decisions = sum(r["decisions"] for r in rows)
    print(
        f"\n{label}: {n} games | rating points per game {statistics.fmean(r['rating_points'] for r in rows):+.3f} | "
        f"mean place {sum(r['place'] for r in rows) / n:.2f} of 6 | "
        f"busted {sum(r['busted_on'] is not None for r in rows)}/{n} | places {sorted(r['place'] for r in rows)} | "
        f"mean chips {sum(r['chips'] for r in rows) / n:.0f} | failed decisions {sum(r['failed'] for r in rows)}/{decisions} | "
        f"cleanup updated {sum(r['cleanup_updated'] for r in rows)}/{sum(r['cleanup_turns'] for r in rows)}"
    )
    for tag in ("button_first_in", "single_open_le_3bb", "bb_vs_late_small_open"):
        counts = [row["preflop_spots"][tag] for row in rows if tag in row["preflop_spots"]]
        if counts:
            print(f"  {tag} folds {sum(c['folds'] for c in counts)}/{sum(c['opportunities'] for c in counts)}")
    for group_name, have_house in (("with house", True), ("all teams", False)):
        group = [row for row in rows if "house_count" in row and bool(row["house_count"]) == have_house]
        if group:
            print(f"  {group_name}: {len(group)} games | {statistics.fmean(r['rating_points'] for r in group):+.3f} "
                  f"points/game | sixths {sum(r['place'] == 6 for r in group)} | "
                  f"busts {sum(r['busted_on'] is not None for r in group)}")


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
    ablations.add_argument("--balanced-play-ablation", action="store_true",
                           help="compare the official prompt with balanced strategy guidance before training")
    ablations.add_argument("--compare-model", help="baseline deployment to pair against --model on the official prompt")
    ablations.add_argument("--guidance-file", type=Path,
                           help="text appended to the model's system prompt, to preview a hint before training on it")
    parser.add_argument("--talk-mode", choices=TALK_MODES, default="normal", help="scripted opponents' table talk")
    parser.add_argument("--lineup", choices=("house", "mixed", "arena"), default="house",
                        help="mixed replaces two house imitations with generic scripted opponents; arena seats "
                             "--field-model in as many seats as other teams' bots took at recent arena tables")
    parser.add_argument("--field-model", default="vals-hackathon/entry",
                        help="deployment(s) that play the other teams' seats with --lineup arena, comma-separated to spread "
                             "the load (default: base Qwen)")
    parser.add_argument("--arena-replays", type=Path, help="with --lineup arena, sample real house rosters from recent public replays")
    parser.add_argument("--arena-entry", default="Friends", help="entry name or org ID for the replay lineups")
    parser.add_argument("--keep-notes", type=Path,
                        help="preview a notes block: keep this text in the model's scratchpad after its first hand, "
                             "as a model trained to keep it would")
    parser.add_argument("--field-no-thinking", action="store_true",
                        help="field seats answer without thinking: a much faster check against weaker opponents")
    args = parser.parse_args()
    if args.games < 1 or args.parallel < 1 or (args.hands is not None and args.hands < 1):
        parser.error("--games, --parallel and --hands must be positive")
    if args.opponent_read_ablation and args.hands == 1:
        parser.error("the opponent-read ablation needs at least two hands")
    if args.arena_replays and args.lineup != "arena":
        parser.error("--arena-replays requires --lineup arena")
    rows = asyncio.run(run(args))
    for arm in dict.fromkeys(row["arm"] for row in rows):
        group = [row for row in rows if row["arm"] == arm]
        print_summary(f"{group[0]['deployment']} ({arm})", group)
    if args.opponent_read_ablation:
        print(json.dumps(compare_arms(rows)))
    elif args.table_talk_defense_ablation:
        print(json.dumps(compare_arms(rows, "table_talk_defense")))
        if args.talk_mode == "false-claims":
            paired = compare_arms(rows, "table_talk_defense", control_arm="specific_claims")
            print(json.dumps({"comparison": "ignore_vs_specific_claims", **paired}))
    elif args.balanced_play_ablation:
        print(json.dumps(compare_arms(rows, "balanced_play")))
    elif args.compare_model:
        print(json.dumps(compare_arms(rows, "candidate")))


if __name__ == "__main__":
    main()
