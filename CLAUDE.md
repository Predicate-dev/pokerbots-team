# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Starter code for the Vals poker event on AC2 (Applied Compute): train `Qwen/Qwen3.6-35B-A3B` to play 6-max no-limit
hold'em, deploy it, enter it in the arena. It is **BYOH-only**: AC2 runs none of this code. The self-hosted harness
(`harness/`) plays and grades every game; AC2 reaches it through `ac2 harness connect`. The root package
`src/pokerbots/` is an intentionally empty stub (AC2 just needs a project version), and `.ac2ignore` keeps everything
else out of the upload. All AC2 tooling defaults to the seeded project `vals-hackathon`; dataset names start with
`pokerbots-`.

Two separate uv projects:

- **Root** (`pyproject.toml`): `vals_tools/`, `rl/`, `opsd/`, `tests/`. Dev deps include `ac2` from a private index;
  if `uv sync` can't find it, export `UV_INDEX_AC2_USERNAME=__token__` and `UV_INDEX_AC2_PASSWORD=<API key>`.
- **Harness** (`harness/pyproject.toml`): the `poker_harness` server and engine (FastAPI, PokerKit). Run its commands
  from `harness/`.

## Commands

```bash
uv sync && (cd harness && uv sync)

uv run pytest                                   # root: tasks, scoring, OPSD builder, training configs
cd harness && uv run pytest -q                  # harness, engine parity, grader
cd harness && uv run pytest tests/test_grading.py::<test_name>   # single test
cd harness && uv run ruff check . && uv run ruff format .        # lint/format (line length 120, py312)

# Serve the harness (keeps running)
cd harness && AC2_API_KEY=<key> uv run uvicorn poker_harness.server:app --host 127.0.0.1 --port 8765
cd harness && SHARDS=4 uv run ./run_sharded.sh  # sharded behind the router on :8765

# Check a deployment plays correctly under the official event rules
cd harness && uv run python play_local.py --model vals-hackathon/<deployment> [--opponents bots] [--hands 6]

# AC2 tooling (all accept --dry-run where they launch something)
uv run python -m vals_tools.doctor --datasets --harness http://127.0.0.1:8765
uv run python rl/make_dataset.py [--dry-run]
uv run python rl/train.py [--dry-run] [--steps 5] [--concurrency 64] [--no-eval]
uv run python opsd/build_dataset.py --eval <eval-id> --dataset pokerbots-opsd-train [--dry-run]
uv run python opsd/train.py --dataset pokerbots-opsd-train [--dry-run]
uv run python -m vals_tools.deploy --name <name> --model <model-or-checkpoint>   # 1 GPU; --delete to free it
uv run python -m vals_tools.eval --dataset pokerbots-smoke --deployment vals-hackathon/<name>
uv run python -m vals_tools.score --eval <eval id>
```

Anything that launches training, deployments or evals spends the team's shared 10-GPU quota (RL run = 8 GPUs, 6 with
`--no-eval`; OPSD = 4; each deployment = 1). Don't launch or delete these without being asked, and never delete the
deployment entered in the arena.

## Architecture

**Rollout flow.** An AC2 training run or eval sends one task (one dataset row, `env_params` = table config) through
the connector to `harness/src/poker_harness/server.py`: `POST /submit` starts a table, `/get_status`,
`/collect_outputs` returns the game log. If the run names no grader, AC2 then calls the Grader API on the same server
(`/submit_grading`, `/get_grading_status`, `/collect_score`), which scores via `grading.py`. Policy seats call the
model through the rollout's AC2 relay URL (so every call is traced); opponent seats call `opponent_model` via the AC2
gateway or `POKER_OPPONENT_BASE_URL`.

**Harness layers** (`harness/src/poker_harness/`):
- `engine/`: hand engine on PokerKit, ported from Vals' engine. Behavior must stay byte-identical:
  `tests/test_engine_parity.py` replays 2,000+ seeded hands against digests in `tests/golden/engine_parity.json`
  (re-record only with `VALS_ENGINE_SRC=... uv run python tests/engine_parity.py --record`).
- `game.py`: plays a whole table (multiple hands, carried stacks, ranking, scratchpad cleanup turns) and builds the
  Vals-shaped game log. Private data (scratchpads, model text) lives only under `log["private"]` and
  `model_response["private"]`.
- `seat.py`: one model seat; each hand is one conversation (assistant turns echoed back so the relay records one
  episode), plus the after-hand `update_scratchpad` cleanup turn.
- `prompts.py`: `TableRules`, system/decision prompts, tool schemas, `sanitize_talk`. `completions.py`: model calls
  and `PRESETS` (`env_params.preset: "event"` = the official event rules; task keys override it).
- `grading.py`: the sample reward, `6 + clip(EV_bb, -5, +5)`, failures score exactly 0. Extension points are
  `hand_reward`, `GRADER_PARAMS`, `artifacts`; keep failures at 0 and real hands above. `ev.py`/`labels.py` compute
  evaluator-only EV labels models never see.
- `tests/golden/talk_off.json` proves table talk off is byte-identical to the pre-talk harness; don't regenerate it.

**Scaling.** Servers hold tables in memory, so never use `uvicorn --workers`; `shard_router.py` routes by
`rollout_id` (or the `Idempotency-Key` header on submit) to one of `SHARDS` single-process servers. The connector
carries at most 1 MiB per rollout, so the returned log is trimmed; full logs go to `harness/runs/`. Restart the
harness only between runs (a running run grades with whatever harness is up).

**Training side.** `vals_tools/tasks.py` defines the task format and the datasets (`pokerbots-train/select/eval/
smoke/table-smoke`); each task is a single fresh 100 BB hand. Opponents are set per dataset/task (`env_params`), not
in `rl/train.py`. `rl/train.py` launches Dr. GRPO (16 hands × 16 attempts per step); `opsd/` does offline OPSD:
`hint.py` defines the hint, `build_dataset.py` turns a base eval's traces into a hinted dataset.

## Hard constraints

The arena always plays the **unmodified** `harness/` with the event rules. A fork of the harness for training is fine,
but the model must still work under the official rules: exactly one `submit_action` call per decision, ≤5,000 output
tokens per decision and ≤10,000 for the cleanup turn (thinking included), and the scratchpad (≤5,000 chars). Verify
with `harness/play_local.py`. Never pass a cluster to AC2 launches (AC2 places jobs itself).
