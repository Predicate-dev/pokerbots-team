"""Test a training run's checkpoints one at a time as they appear: deploy, evaluate, compare, delete, next.

    AC2_API_KEY=<key> uv run python checkpoint_loop.py --train train_xxx --baseline v4 \
        --notes-file ../opsd/notes/call_checklist.txt --prefix grpo3

One spare GPU is enough. For each new checkpoint, lowest step first, the loop

1. deploys it (`vals_tools.deploy`, 1 GPU),
2. runs the quick part of `eval_suite.py` on it (held-out call decisions with the notes and with empty notes),
3. compares it with the baseline's saved results, paired by decision, and appends one line to
   `runs/suite/<prefix>_checkpoints.jsonl`,
4. deletes the deployment, so the GPU is free for the next checkpoint.

It stops when the run has ended and every checkpoint has been tested, then names the best one. Nothing is swapped
or stopped automatically: a checkpoint that is clearly worse than the baseline is flagged, and the best one still
has to pass the full suite (`eval_suite.py run` with games) before anyone enters it. Checkpoints themselves stay on
AC2 (they cost storage, not GPUs), so a step can always be redeployed later.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import eval_suite

ROOT = Path(__file__).resolve().parents[1]
ROOT_PYTHON = str(ROOT / ".venv" / "bin" / "python")
LIST = """
import json, sys
from ac2 import Client
c = Client(project=sys.argv[2])
print(json.dumps({"status": str(getattr(c.train.get(sys.argv[1]), "status", "")),
                  "checkpoints": [[getattr(x, "iter_num", None), x.id] for x in c.checkpoints.list(train_job_id=sys.argv[1])]}))
"""


def state(train: str, project: str) -> dict:
    out = subprocess.run([ROOT_PYTHON, "-c", LIST, train, project], capture_output=True, text=True, check=False, cwd=ROOT)
    try:
        return json.loads(out.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"status": "unknown", "checkpoints": []}


def deploy(name: str, checkpoint: str | None) -> bool:
    command = [ROOT_PYTHON, "-u", "-m", "vals_tools.deploy", "--name", name]
    command += ["--checkpoint", checkpoint] if checkpoint else ["--delete"]
    done = subprocess.run(command, capture_output=True, text=True, check=False, cwd=ROOT)
    return done.returncode == 0


def compare(name: str, baseline: str) -> dict:
    candidate, base = eval_suite.load(name), eval_suite.load(baseline)
    row: dict = {"name": name}
    for key in ("calls_notes", "calls_empty"):
        mine = list(candidate[key].values())
        if not mine:
            continue
        row[key] = {
            "points_per_1000": round(1000 * sum(eval_suite.gain(r) for r in mine) / len(mine), 1),
            "invalid": sum(r["answer"] == "invalid" for r in mine), "decisions": len(mine),
        }
        both = sorted(set(candidate[key]) & set(base[key]))
        if both:
            mean, low, high = eval_suite.bootstrap([eval_suite.gain(candidate[key][i]) - eval_suite.gain(base[key][i]) for i in both], 1000)
            row[key] |= {"vs_baseline": round(mean, 1), "low": round(low, 1), "high": round(high, 1)}
    return row


def line(step: int, row: dict) -> str:
    parts = [f"step {step:>3}"]
    for key, label in (("calls_notes", "with notes"), ("calls_empty", "empty notes")):
        r = row.get(key)
        if r:
            versus = f", vs baseline {r['vs_baseline']:+.1f} [{r['low']:+.0f}, {r['high']:+.0f}]" if "vs_baseline" in r else ""
            parts.append(f"{label}: {r['points_per_1000']:+.1f} per 1000{versus}, invalid {r['invalid']}/{r['decisions']}")
    return " | ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", required=True, help="the training run's id")
    parser.add_argument("--baseline", required=True, help="a results folder under runs/suite/ to compare with")
    parser.add_argument("--notes-file", type=Path, required=True)
    parser.add_argument("--prefix", required=True, help="deployment and results name prefix")
    parser.add_argument("--project", default="vals-hackathon")
    parser.add_argument("--skip-steps", type=int, nargs="*", default=[], help="checkpoint steps not to test")
    parser.add_argument("--parallel", type=int, default=16)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    table = eval_suite.SUITE / f"{args.prefix}_checkpoints.jsonl"
    table.parent.mkdir(parents=True, exist_ok=True)
    tested: dict[int, dict] = {}
    if table.exists():
        for text in table.read_text().splitlines():
            row = json.loads(text)
            tested[row["step"]] = row
    while True:
        now = state(args.train, args.project)
        waiting = sorted((step, ckpt) for step, ckpt in now["checkpoints"] if step not in tested and step not in args.skip_steps)
        if not waiting:
            if now["status"] not in ("running", "queued", "pending", "starting", "unknown"):
                break
            time.sleep(args.poll_seconds)
            continue
        step, checkpoint = waiting[0]
        name = f"{args.prefix}-s{step}"
        print(f"{time.strftime('%H:%M')} step {step} ({checkpoint}): deploying as {name}", flush=True)
        if not deploy(name, checkpoint):
            print(f"  deploy failed; will retry on the next pass", flush=True)
            time.sleep(args.poll_seconds)
            continue
        suite = subprocess.run(
            [sys.executable, "eval_suite.py", "run", "--model", f"{args.project}/{name}", "--name", name,
             "--notes-file", str(args.notes_file), "--parallel", str(args.parallel), "--skip", "games", "notes", "opens"],
            capture_output=True, text=True, check=False,
        )
        deploy(name, None)  # free the GPU whatever happened
        if suite.returncode:
            print(f"  evaluation failed:\n{suite.stdout[-400:]}", flush=True)
            tested[step] = {"step": step, "checkpoint": checkpoint, "failed": True}
            continue
        row = compare(name, args.baseline) | {"step": step, "checkpoint": checkpoint}
        tested[step] = row
        with table.open("a") as out:
            out.write(json.dumps(row) + "\n")
        notes = row.get("calls_notes", {})
        flag = ""
        if notes.get("high", 0) < 0:
            flag = "   <-- clearly worse than the baseline"
        elif notes.get("low", 0) > 0:
            flag = "   <-- clearly better than the baseline"
        if notes.get("decisions") and notes["invalid"] > 0.05 * notes["decisions"]:
            flag += "   <-- too many invalid replies"
        print(f"{time.strftime('%H:%M')} {line(step, row)}{flag}", flush=True)
    scored = {step: row for step, row in tested.items() if row.get("calls_notes", {}).get("vs_baseline") is not None}
    print("\nall checkpoints tested:")
    for step in sorted(scored):
        print("  " + line(step, scored[step]))
    if scored:
        best = max(scored, key=lambda step: scored[step]["calls_notes"]["vs_baseline"])
        print(f"best on calls with notes: step {best} ({scored[best]['checkpoint']}). Run the full suite with games on it before entering it.")


if __name__ == "__main__":
    main()
