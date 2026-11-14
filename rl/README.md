# RL: GRPO on single 6-max hands

[`train.py`](train.py) trains Qwen3.6-35B-A3B on `pokerbots-train` through your self-hosted harness:

- **Rollouts:** each hand is played by the harness; the trained seat calls the run's inference engine through the
  rollout relay and the five opponents call your opponent endpoint.
- **Reward:** harness-side grading, `6 + clip(EV / 1 BB, -5, +5)`, failures 0 (see the [main README](../README.md)).
  The run names no grader.
- **Method:** Dr. GRPO (`--advantage dr-grpo`) with a small KL penalty (`--kl-coef 0.01`), 16 hands per step
  (`--hands-per-step`) with 16 attempts each (`--samples-per-hand`), 60 steps (`--steps`).
- **Replicas:** 1 rollout and 1 eval replica (`--inference-replicas`, `--eval-replicas`). These are the training
  run's own engines, 2 GPUs each (deployments are separate, 1 GPU each), so the run uses about 8 GPUs in all.
  AC2's own default for this model is 4 of each, 20 GPUs, more than a team's 10-GPU quota.
- **Token budget:** see below. **Everything else is AC2's default** (learning rate, sampling concurrency,
  checkpoints kept) unless you pass the matching flag. `--dry-run` lists every setting the launch sends,
  then the exact configuration.
- **Scratchpad:** tasks use the event preset, so every rollout ends with the after-hand `update_scratchpad` turn.
  In a single-hand task that update is never read, and the grader ignores it.
- **Token budget:** `--max-response-len 24576` (and `--max-total-len 32768`). In training this is a budget for the
  whole hand: every reply's reasoning, every observation and the cleanup turn. AC2's default (8,192) runs out after a
  couple of long decisions, and the rest of the hand is played as check or fold. Raise it for longer hands.
- **Checkpoints and eval:** a checkpoint every 10 steps (`--checkpoint-every`), and an in-run eval on
  `pokerbots-select` at step 0 (base) and every 10 steps (`--eval-every`, or `--no-eval`).

## Training for finishing place, not chips

The arena rates a game on where you finish, not on chips: first by 1 chip equals first by 400, and a busted seat
ranks below every seat still in. The default reward can't see that. It scores one fresh first hand on chips, capped
at 5 BB either way, so losing a whole stack costs the same as losing 5 BB and the model never plays a hand late in
a game. The finishing-place datasets fix both:

```bash
uv run python rl/make_dataset.py --placement --prefix pokerbots-place --dry-run
uv run python rl/make_dataset.py --placement --prefix pokerbots-place
uv run python rl/train.py --prefix pokerbots-place --dry-run
```

- **Half the tasks are mid-game hands.** The model is dropped into one hand of a 30-hand game ("Hand 27 of 30")
  with uneven stacks and sometimes a seat already out (`env_params.starting_stacks`, `first_hand`, `play_hands`).
  Half of those are hand 24 or later. The stacks are made up (`vals_tools/tasks.py`, `midgame_stacks`), not sampled
  from real games, and the scratchpad still starts blank.
- **Those hands are scored on finishing place** (`grader_params.reward: "placement"`): the change in the share of
  the table the seat is expected to finish above, `6 + clip(10 x change, -5, +5)`. Going from an even table to
  certain last is -5; the same chips are worth more the fewer hands are left. See `harness/.../placement.py`.
- **The other half are the usual fresh first hands, scored on chips,** where the two rewards agree and chips give
  GRPO a stronger signal.
- Restart the harness after pulling this change, and only between runs. The arena still plays the official rules:
  these settings exist only in training tasks.

## Run

With the harness and connector running and the datasets uploaded:

```bash
uv run python rl/train.py --project <project> --dry-run     # print the resolved config
uv run python -m vals_tools.eval --project <project> --dataset pokerbots-smoke --model <catalog-model>   # harness smoke first
uv run python rl/train.py --project <project>
```

`--concurrency` caps concurrent hands on your harness. It's unset by default, so AC2 may send a whole step (16 hands
× 16 attempts = 256 games) at once; `--concurrency 64` suits the default `SHARDS=4`. `--steps`, `--lr` and `--prefix`
change the obvious things.

## Cost and capacity

- **GPUs:** about 8 while running: 4 to train, 2 for the rollout engine, 2 for the in-run eval engine (each of AC2's
  training inference engines uses 2 GPUs for this model; deployments use 1).
- **Opponent calls:** every hand makes several opponent calls per seat, so a 256-hand step is well over a thousand
  opponent calls. The provided model allows about 60 requests per second for your org; check your own endpoint's
  price and rate limits before a long run.
- **Harness:** one shard handles a few dozen concurrent hands. Run enough shards (`SHARDS=`) for the run's
  concurrency plus its sidecar eval.

## Judge it

1. Pick a checkpoint from the sidecar on `-select`, compared with step 0.
2. Serve that checkpoint and base the same way (as inference deployments) and evaluate both on
   `pokerbots-eval`.
3. `uv run python -m vals_tools.score --project <project> --eval <checkpoint-eval> --base <base-eval>`.
   The difference from base is paired per deal; its 95% CI tells you whether it is more than noise.
