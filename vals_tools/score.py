"""Score poker EV and placement, paired by task against a base eval on the same dataset.

    uv run python -m vals_tools.score --project <project> --eval eval_new --base eval_base
    uv run python -m vals_tools.score --project <project> --run train_xxx --step 20

Reads the grader's result artifacts. EV totals are normalized by hands played; paired intervals use tasks as
the independent units. Placement comparisons count games and report places gained versus base.
"""

from __future__ import annotations

import argparse
import json
import statistics
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ac2.sdk import Client

COLUMNS = ("job_id", "generated_step", "datapoint_id", "artifacts", "parent_score_id", "created_at")


def score_rows(client: Client, eval_id: str | None = None, run: str | None = None, step: int | None = None) -> dict:
    """Top-level harness scores keyed by task id; the latest score wins on retries."""
    rows: dict[str, dict] = {}
    query = client.analytics.scores(run_ids=[eval_id or run]).select(*COLUMNS)
    for row in sorted(query.iter_rows(), key=lambda r: str(r["created_at"])):
        if row["parent_score_id"]:
            continue
        job = str(row["job_id"])
        if eval_id and job != eval_id:
            continue
        if run and (not job.startswith("eval_") or str(row["generated_step"]) != str(step)):
            continue
        artifacts = row["artifacts"] if isinstance(row["artifacts"], dict) else json.loads(row["artifacts"] or "{}")
        rows[str(row["datapoint_id"])] = artifacts
    return rows


def _ci95(values: list[float]) -> float | None:
    return round(1.96 * statistics.stdev(values) / len(values) ** 0.5, 3) if len(values) > 1 else None


def _hands(row: dict) -> int:
    return row.get("hands", 1)


def _ev_rate(row: dict) -> float:
    return row["ev_bb"] / _hands(row)


def summarize(rows: dict, base: dict | None = None) -> dict[str, Any]:
    ev = {key: row for key, row in rows.items() if "ev_bb" in row and _hands(row) > 0}
    placement = {key: row for key, row in rows.items() if "place" in row and _hands(row) > 0}
    scored = ev | placement
    hands = sum(_hands(row) for row in scored.values())
    out: dict[str, Any] = {
        "games": len(scored),
        "hands": hands,
        "unscored_tasks": len(rows) - len(scored),
    }
    if scored:
        failed = sum(row["policy_format_errors"] > 0 for row in scored.values()) / len(scored)
        out["format_error_games"] = round(failed, 4)
        if hands == len(scored):
            out["format_error_hands"] = out["format_error_games"]
    if ev:
        ev_hands = sum(_hands(row) for row in ev.values())
        ev_rate = sum(row["ev_bb"] for row in ev.values()) / ev_hands
        # Interval around the pooled rate, treating each rollout as a cluster.
        average_hands = ev_hands / len(ev)
        residuals = [(row["ev_bb"] - ev_rate * _hands(row)) / average_hands for row in ev.values()]
        out.update(ev_bb_per_hand=round(ev_rate, 3), ev_bb_per_100=round(100 * ev_rate, 3), ci95=_ci95(residuals))
    if placement:
        places = [row["place"] for row in placement.values()]
        out.update(mean_place=round(statistics.fmean(places), 3), place_ci95=_ci95(places),
                   bust_rate=round(statistics.fmean(row["busted"] for row in placement.values()), 4))
    if base is not None:
        out["pairing"] = {
            "candidate_only": len(rows.keys() - base.keys()),
            "base_only": len(base.keys() - rows.keys()),
        }
        paired_ev = [key for key in ev.keys() & base.keys() if "ev_bb" in base[key] and _hands(base[key]) > 0]
        diffs = [_ev_rate(ev[key]) - _ev_rate(base[key]) for key in paired_ev]
        if diffs:
            out["vs_base"] = {
                "paired_games": len(diffs),
                "paired_hands": sum(_hands(ev[key]) for key in paired_ev),
                "diff": round(statistics.fmean(diffs), 3),
                "ci95": _ci95(diffs),
            }
        place_diffs = [
            base[key]["place"] - placement[key]["place"] for key in placement.keys() & base.keys() if "place" in base[key]
        ]
        if place_diffs:
            out["placement_vs_base"] = {
                "paired_games": len(place_diffs),
                "mean_places_gained": round(statistics.fmean(place_diffs), 3),
                "ci95": _ci95(place_diffs),
            }
    return out


def main() -> None:
    from ac2.sdk import Client

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default="vals-hackathon")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--eval", help="a standalone eval")
    target.add_argument("--run", help="a training run; with --step, its sidecar eval at that step")
    parser.add_argument("--step", type=int)
    parser.add_argument("--base", help="the base model's eval on the same dataset")
    args = parser.parse_args()

    client = Client(project=args.project)
    rows = score_rows(client, eval_id=args.eval, run=args.run, step=args.step)
    base = score_rows(client, eval_id=args.base) if args.base else None
    print(json.dumps(summarize(rows, base), indent=1))


if __name__ == "__main__":
    main()
