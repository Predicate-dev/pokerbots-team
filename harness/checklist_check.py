"""Does a deployment write the call checklist into its scratchpad and keep it there through a game?

    AC2_API_KEY=<key> uv run python checklist_check.py --model vals-hackathon/<deployment> --games 4 --hands 10

Plays short event-rules games against scripted bots (`play_local.py --opponents bots`), then reads the model's
scratchpad after every hand from the game logs. It reports after which hand the checklist first appeared, in how
many later hands it was still there word for word, how long the notes got, and how many decisions and scratchpad
updates failed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BLOCK = (Path(__file__).resolve().parents[1] / "opsd" / "notes" / "call_checklist.txt").read_text().strip()


def play(model: str, hands: int, seed: int, out: Path) -> Path | None:
    done = subprocess.run(
        [sys.executable, "play_local.py", "--model", model, "--opponents", "bots", "--hands", str(hands),
         "--seed", str(seed), "--out", str(out)],
        capture_output=True, text=True, check=False,
    )
    for line in done.stdout.splitlines():
        if "log:" in line:
            return Path(line.split("log:")[1].strip())
    print(f"seed {seed}: no log\n{done.stdout[-400:]}\n{done.stderr[-400:]}")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--hands", type=int, default=10)
    parser.add_argument("--seed", type=int, default=4100)
    parser.add_argument("--out", type=Path, default=Path("runs/checklist_check"))
    args = parser.parse_args()
    with ThreadPoolExecutor(args.games) as pool:
        logs = list(pool.map(lambda s: play(args.model, args.hands, s, args.out), range(args.seed, args.seed + args.games)))
    total = {"hands": 0, "exact": 0, "title": 0, "decisions": 0, "failed": 0}
    for path in logs:
        if path is None:
            continue
        log = json.loads(path.read_text())
        mine = [a for hand in log["hands"] for a in hand["actions"] if a["player"] == "seat0"]
        failed = sum(bool(a["model_response"].get("format_error")) for a in mine)
        hands = log["private"]["seat0"]["hands"]
        first, exact, title, lengths = None, 0, 0, []
        for number, record in sorted(hands.items(), key=lambda kv: int(kv[0])):
            after = record.get("scratchpad_after")
            if after is None:
                continue
            lengths.append(len(after))
            exact += after.startswith(BLOCK)
            title += "CALL CHECKLIST" in after
            if first is None and "CALL CHECKLIST" in after:
                first = int(number)
        total["hands"] += len(lengths)
        total["exact"] += exact
        total["title"] += title
        total["decisions"] += len(mine)
        total["failed"] += failed
        print(f"{path.name}: checklist first written after hand {first} | word for word after {exact}/{len(lengths)} hands | "
              f"present in some form {title}/{len(lengths)} | notes length {min(lengths, default=0)}-{max(lengths, default=0)} | "
              f"failed decisions {failed}/{len(mine)} | net {log['final_stacks']['seat0'] - 200:+d} chips")
    print(f"\n{args.model}: checklist word for word after {total['exact']}/{total['hands']} hands, present in some form "
          f"{total['title']}/{total['hands']}; failed decisions {total['failed']}/{total['decisions']}")


if __name__ == "__main__":
    main()
