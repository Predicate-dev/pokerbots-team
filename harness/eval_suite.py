"""The whole offline evaluation of a deployment, run at once, and a paired comparison of two deployments.

    # every test concurrently against one deployment; results land in runs/suite/<name>/
    AC2_API_KEY=<key> uv run python eval_suite.py run --model vals-hackathon/<deployment> --name v5 \
        --notes-file ../opsd/notes/call_checklist_v2.txt --field-model vals-hackathon/<field> --games 48

    # candidate against baseline, paired wherever both saw the same decision or the same deal
    uv run python eval_suite.py report --candidate v5 --baseline v4

What `run` starts, all at the same time:

- calls with notes  held-out call decisions with the notes block already in the scratchpad (`opsd/price_run.py`)
- calls, empty      the same decisions with an empty scratchpad, which is every game's first hand
- opens             held-out first-in raise-or-fold decisions against the open card (`--decision open`)
- notes             short real games: are the notes written after hand 1 and kept word for word (`checklist_check.py`)
- games             full 30-hand arena-like games on fixed seeds (`score_local.py`)

`report` reads two result folders. Call decisions are paired by decision id and games by seed, and each difference
comes with a 95% bootstrap range, because a handful of large decisions and a few last places dominate both scores.
A difference whose range includes zero is not evidence of anything: sixteen games cannot separate two models that
differ by 0.1 rating points per game, so play at least 48 before believing a game result.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "opsd"))
POINTS = {1: 0.83, 2: 0.63, 3: 0.38, 4: 0.05, 5: -0.45, 6: -1.45}  # rating change by finishing place, six equal seats
SUITE = Path("runs/suite")


def gain(row: dict) -> float:
    """Rating points a call answer is worth over folding, from the spot's real equity (as `opsd/price_run.py`)."""
    c = row["card"]
    if row["answer"] != "continue" or row.get("equity") is None:
        return 0.0
    return row["equity"] * c["win_value"] + (1 - row["equity"]) * c["lose_value"] - c["fold_value"]


def bootstrap(values: list[float], scale: float = 1.0, rounds: int = 2000) -> tuple[float, float, float]:
    """(mean, low, high) of a list, with a 95% bootstrap range."""
    if not values:
        return 0.0, 0.0, 0.0
    rng = random.Random(0)
    n = len(values)
    means = sorted(scale * sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(rounds))
    return scale * sum(values) / n, means[int(0.025 * rounds)], means[int(0.975 * rounds) - 1]


# ---------------------------------------------------------------------------------------------------------- run


async def start(label: str, command: list[str], log: Path, cwd: Path) -> tuple[str, int, float]:
    began = time.monotonic()
    with log.open("w") as out:
        process = await asyncio.create_subprocess_exec(*command, stdout=out, stderr=asyncio.subprocess.STDOUT, cwd=cwd)
        code = await process.wait()
    took = time.monotonic() - began
    print(f"  {label}: {'done' if code == 0 else f'FAILED (exit {code}, see {log})'} in {took / 60:.1f} min", flush=True)
    return label, code, took


async def run(args: argparse.Namespace) -> None:
    out = (SUITE / args.name).resolve()
    out.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent
    python = sys.executable
    notes = str(args.notes_file.resolve()) if args.notes_file else None
    spots, open_spots = str(args.spots.resolve()), args.open_spots.resolve()
    price = [python, str(ROOT / "opsd" / "price_run.py"), "--model", args.model, "--held-out", "--parallel", str(args.parallel)]
    jobs: dict[str, list[str]] = {}
    if "calls" not in args.skip:
        jobs["calls, empty notes"] = [*price, "--spots", spots, "--out", str(out / "calls_empty.jsonl")]
        if notes:
            jobs["calls with notes"] = [*price, "--spots", spots, "--notes-file", notes, "--out", str(out / "calls_notes.jsonl")]
    if "opens" not in args.skip and open_spots.exists():
        jobs["opens"] = [*price, "--decision", "open", "--spots", str(open_spots), "--limit", str(args.open_limit),
                         "--out", str(out / "opens.jsonl"),
                         *(["--notes-file", notes] if notes else [])]
    if "notes" not in args.skip and notes:
        jobs["notes"] = [python, "checklist_check.py", "--model", args.model, "--games", "4", "--hands", "10",
                         "--notes-file", notes, "--out", str(out / "notes_games")]
    if "games" not in args.skip and args.games:
        jobs["games"] = [python, "score_local.py", "--model", args.model, "--lineup", "arena", "--field-model", args.field_model,
                         *(["--field-no-thinking"] if not args.field_thinking else []),
                         *(["--arena-replays", str(args.arena_replays.resolve()), "--arena-entry", args.arena_entry]
                           if args.arena_replays else []),
                         "--games", str(args.games), "--seed", str(args.seed),
                         "--parallel", str(args.game_parallel), "--out", str(out / "games")]
    (out / "meta.json").write_text(json.dumps({"model": args.model, "notes_file": notes, "field_model": args.field_model,
                                               "games": args.games, "seed": args.seed, "jobs": list(jobs)}, indent=1))
    print(f"{args.model}: {len(jobs)} tests started at once; results in {out}", flush=True)
    results = await asyncio.gather(*(start(label, command, out / f"{label.replace(', ', '_').replace(' ', '_')}.log", here)
                                     for label, command in jobs.items()))
    failed = [label for label, code, _ in results if code]
    print()
    show(args.name, load(args.name))
    if failed:
        raise SystemExit(f"failed: {', '.join(failed)}")


# ------------------------------------------------------------------------------------------------------- results


def rows(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {r["id"]: r for r in map(json.loads, path.read_text().splitlines()) if r.get("card")}


def games(folder: Path) -> dict[int, dict]:
    """One record per finished game log under the folder (any depth), keyed by seed."""
    out = {}
    for path in (Path(top) / name for top, _, names in os.walk(folder, followlinks=True) for name in names if name.endswith(".json")):
        try:
            log = json.loads(path.read_text())
        except ValueError:
            continue
        seat = log.get("scored_seat")
        if not log.get("complete") or seat is None or "ranking" not in log:
            continue
        mine = [a for hand in log["hands"] for a in hand["actions"] if a["player"] == seat]
        out[int(log["session_info"]["seed"])] = {
            "place": log["ranking"]["places"][seat], "busted": log["eliminated_on_hand"].get(seat) is not None,
            "decisions": len(mine), "failed": sum(bool(a["model_response"].get("format_error")) for a in mine),
            "house_count": len(log["house_opponents"]) if "house_opponents" in log else None,
            "matchup": log.get("matchup") or (seat, log.get("lineup")),
        } | habits(log, seat)
    return out


def habits(log: dict, seat: str) -> dict:
    """The habits that go with finishing in the top three at today's tables (see the arena behaviour analysis):
    small pots taken without a showdown, raising when first in, betting the flop after raising, and no big loss."""
    h = dict.fromkeys(("chances", "raises", "limps", "uncontested", "flop_spots", "flop_bets", "faced_open", "reraises", "big_loss"), 0)
    for hand in log["hands"]:
        if seat not in hand["starting_state"]:
            continue
        acts = hand["actions"]
        pre = [a for a in acts if a["street"] == "preflop"]
        first = next((a for a in pre if a["player"] == seat), None)
        opened = False
        if first is not None:
            before = pre[: pre.index(first)]
            kind = first["action"]["type"]
            if all(a["action"]["type"] == "fold" for a in before) and first.get("call_amount", 0) > 0:
                h["chances"] += 1
                h["raises"] += kind == "raise"
                h["limps"] += kind == "call"
                opened = kind == "raise"
            elif sum(a["action"]["type"] == "raise" for a in before) == 1:
                h["faced_open"] += 1
                h["reraises"] += kind == "raise"
        flop = [a for a in acts if a["street"] == "flop" and a["player"] == seat]
        if opened and flop and flop[0].get("call_amount", 0) == 0:
            h["flop_spots"] += 1
            h["flop_bets"] += flop[0]["action"]["type"] in ("bet", "raise")
        change = hand["ending_stacks"].get(seat, 0) - hand["starting_state"][seat]["starting_stack"]
        h["uncontested"] += change > 0 and not hand.get("showdown")
        h["big_loss"] = max(h["big_loss"], -change)
    return h


def notes_line(folder: Path) -> dict | None:
    for path in sorted(folder.glob("notes*")):
        if path.is_file():
            found = re.search(r"word for word after (\d+)/(\d+) hands, present in some form (\d+)/(\d+); failed decisions (\d+)/(\d+)",
                              path.read_text())
            if found:
                return dict(zip(("exact", "hands", "present", "hands2", "failed", "decisions"), map(int, found.groups()), strict=True))
    return None


def load(name: str) -> dict:
    folder = SUITE / name
    if not folder.exists():
        raise SystemExit(f"no results in {folder}")
    return {"calls_notes": rows(folder / "calls_notes.jsonl"), "calls_empty": rows(folder / "calls_empty.jsonl"),
            "opens": rows(folder / "opens.jsonl"), "notes": notes_line(folder), "games": games(folder)}


def calls_summary(data: dict[str, dict]) -> str:
    r = list(data.values())
    mean, low, high = bootstrap([gain(x) for x in r], 1000)
    best = sum((x["answer"] == "continue") == (x["equity"] * x["card"]["win_value"] + (1 - x["equity"]) * x["card"]["lose_value"]
                                                 > x["card"]["fold_value"]) for x in r)
    clear = [x for x in r if x["card"]["plan"] != "close"]
    return (f"{len(r)} decisions | {mean:+.1f} rating pts per 1000 [{low:+.0f}, {high:+.0f}] | best answer {100 * best / len(r):.0f}% | "
            f"follows the card {100 * sum(x['answer'] == x['card']['plan'] for x in clear) / max(1, len(clear)):.0f}% | "
            f"continues {100 * sum(x['answer'] == 'continue' for x in r) / len(r):.0f}% | invalid {sum(x['answer'] == 'invalid' for x in r)} | "
            f"median reply {statistics.median((x.get('out_tokens') or 0) for x in r):.0f} tokens")


def games_summary(data: dict[int, dict]) -> str:
    g = list(data.values())
    mean, low, high = bootstrap([POINTS[x["place"]] for x in g])
    places = [sum(x["place"] == p for x in g) for p in range(1, 7)]
    top3 = 100 * sum(places[:3]) / len(g)
    line = (f"{len(g)} games | {mean:+.3f} rating pts per game [{low:+.2f}, {high:+.2f}] | top three {top3:.0f}% | places 1st-6th {places} | "
            f"busted {sum(x['busted'] for x in g)} | failed decisions {sum(x['failed'] for x in g)}/{sum(x['decisions'] for x in g)}")
    if "chances" not in g[0]:
        return line
    total = {k: sum(x[k] for x in g) for k in ("chances", "raises", "limps", "uncontested", "flop_spots", "flop_bets", "faced_open", "reraises")}
    return line + (
        f"\n  habits           : first-in raises {total['raises'] / len(g):.1f} per game ({100 * total['raises'] / max(1, total['chances']):.0f}% of chances; "
        f"just calls {100 * total['limps'] / max(1, total['chances']):.0f}%) | pots won without a showdown {total['uncontested'] / len(g):.1f} per game | "
        f"bets the flop after raising {total['flop_bets']}/{total['flop_spots']} | re-raises an open {total['reraises']}/{total['faced_open']} | "
        f"games with a 21+ chip loss {100 * sum(x['big_loss'] >= 21 for x in g) / len(g):.0f}%")


def show(name: str, data: dict) -> None:
    print(f"== {name}")
    if data["calls_notes"]:
        print("  calls with notes :", calls_summary(data["calls_notes"]))
    if data["calls_empty"]:
        print("  calls, empty     :", calls_summary(data["calls_empty"]))
    if data["opens"]:
        r = list(data["opens"].values())
        clear = [x for x in r if x["card"].get("clear")]
        print(f"  opens            : {len(r)} decisions | follows clear open-card plans {sum(x['answer'] == x['card']['plan'] for x in clear)}/{len(clear)} | "
              f"raises {100 * sum(x['answer'] == 'raise' for x in r) / len(r):.0f}% | invalid {sum(x['answer'] == 'invalid' for x in r)}")
    if data["notes"]:
        n = data["notes"]
        print(f"  notes            : word for word after {n['exact']}/{n['hands']} hands | present {n['present']}/{n['hands']} | "
              f"failed decisions {n['failed']}/{n['decisions']}")
    if data["games"]:
        print("  games            :", games_summary(data["games"]))
        with_house = [row for row in data["games"].values() if (row["house_count"] or 0) > 0]
        teams_only = [row for row in data["games"].values() if row["house_count"] == 0]
        for label, group in (("with house", with_house), ("all teams", teams_only)):
            if group:
                print(f"    {label:12} {len(group):3} games | {statistics.fmean(POINTS[row['place']] for row in group):+.3f} "
                      f"pts/game | sixths {sum(row['place'] == 6 for row in group)} | busts {sum(row['busted'] for row in group)}")


def report(args: argparse.Namespace) -> None:
    candidate, baseline = load(args.candidate), load(args.baseline)
    show(args.baseline, baseline)
    show(args.candidate, candidate)
    print(f"\n== {args.candidate} minus {args.baseline}, paired (a range that includes 0 is not evidence)")
    verdicts = []
    for key, label in (("calls_notes", "calls with notes"), ("calls_empty", "calls, empty")):
        both = sorted(set(candidate[key]) & set(baseline[key]))
        if both:
            mean, low, high = bootstrap([gain(candidate[key][i]) - gain(baseline[key][i]) for i in both], 1000)
            print(f"  {label:17}: {mean:+.1f} rating pts per 1000 [{low:+.0f}, {high:+.0f}] over {len(both)} shared decisions")
            verdicts.append((label, low, high))
    shared = set(candidate["games"]) & set(baseline["games"])
    seeds = sorted(s for s in shared if candidate["games"][s]["matchup"] == baseline["games"][s]["matchup"])
    if len(shared) != len(seeds):
        print(f"  skipped {len(shared) - len(seeds)} games with different opponent lineups or settings")
    if seeds:
        diffs = [POINTS[candidate["games"][s]["place"]] - POINTS[baseline["games"][s]["place"]] for s in seeds]
        mean, low, high = bootstrap(diffs)
        better = sum(candidate["games"][s]["place"] < baseline["games"][s]["place"] for s in seeds)
        worse = sum(candidate["games"][s]["place"] > baseline["games"][s]["place"] for s in seeds)
        t_mean, t_low, t_high = bootstrap([(candidate["games"][s]["place"] <= 3) - (baseline["games"][s]["place"] <= 3) for s in seeds], 100)
        print(f"  games            : {mean:+.3f} rating pts per game [{low:+.2f}, {high:+.2f}] over {len(seeds)} shared deals | "
              f"better {better}, same {len(seeds) - better - worse}, worse {worse} | top-three rate {t_mean:+.0f} points [{t_low:+.0f}, {t_high:+.0f}]")
        for label, group in (
            ("with house", [s for s in seeds if (candidate["games"][s]["house_count"] or 0) > 0]),
            ("all teams", [s for s in seeds if candidate["games"][s]["house_count"] == 0]),
        ):
            if group:
                effect, a, b = bootstrap([POINTS[candidate["games"][s]["place"]] - POINTS[baseline["games"][s]["place"]] for s in group])
                print(f"    {label:12} {effect:+.3f} points/game [{a:+.2f}, {b:+.2f}] over {len(group)} paired games")
        verdicts.append(("games", low, high))
    for label, low, high in verdicts:
        print(f"  -> {label}: {'better' if low > 0 else 'worse' if high < 0 else 'no clear difference'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    r = commands.add_parser("run", help="run every test against one deployment, concurrently")
    r.add_argument("--model", required=True)
    r.add_argument("--name", required=True, help="results folder under runs/suite/")
    r.add_argument("--notes-file", type=Path, help="the block this model keeps at the top of its scratchpad")
    r.add_argument("--spots", type=Path, default=Path("runs/price_spots_v2.jsonl"))
    r.add_argument("--open-spots", type=Path, default=Path("runs/open_spots.jsonl"))
    r.add_argument("--open-limit", type=int, default=600, help="how many held-out opening decisions to ask")
    r.add_argument("--field-model", default="vals-hackathon/v2-field", help="the deployment in the other model seats of the games")
    r.add_argument("--field-thinking", action="store_true", help="enable thinking for the model opponents")
    r.add_argument("--arena-replays", type=Path, help="use the house lineup and seats from public replay samples")
    r.add_argument("--arena-entry", default="Friends", help="source entry name or org ID for replay lineups")
    r.add_argument("--games", type=int, default=48)
    r.add_argument("--seed", type=int, default=20261300)
    r.add_argument("--parallel", type=int, default=12, help="requests at once for each decision test")
    r.add_argument("--game-parallel", type=int, default=8, help="games at once")
    r.add_argument("--skip", nargs="*", default=[], choices=["calls", "opens", "notes", "games"])
    p = commands.add_parser("report", help="compare two result folders")
    p.add_argument("--candidate", required=True)
    p.add_argument("--baseline", required=True)
    args = parser.parse_args()
    if args.command == "run":
        asyncio.run(run(args))
    else:
        report(args)


if __name__ == "__main__":
    main()
