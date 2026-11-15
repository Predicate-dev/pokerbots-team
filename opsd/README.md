# Offline OPSD: distill a strategy hint

At the event the harness owns the system prompt, so any strategy you want the model to follow has to be in its
weights. Offline OPSD (on-policy self-distillation) puts it there: the teacher is the base model with a hint in its
system prompt, the student is the same model without it, and training pulls the student's token distribution
toward the teacher's on replies the base model actually wrote.

The default teacher uses position, effective stack, pot odds, and remaining hands. Select `--strategy tight-survival`
to reproduce the original hint in [`hint.py`](hint.py).

## Test the balanced strategy

The guidance applies position-sensitive opening, big-blind defense against small raises, and placement adjustments.
For a big blind facing a single open up to 3 BB with two seats left in the hand and chips behind, the teacher compares
the price with simulated check-down equity against three blocker-adjusted opening ranges. Each range adds combinations
to the previous one. This estimates how much equity the call would need to realize after future betting. Its range
widths are illustrative and the primary score for the full game is still finishing place.
[Pluribus](https://noambrown.com/papers/19-Science-Superhuman.pdf),
[Computing Robust Counter-Strategies](https://poker.cs.ualberta.ca/publications/NIPS07-rnash.pdf).

First compare the hint and the official prompt using the same model, deck seeds, seats, and lineups. Then repeat
with unused seeds and another lineup.

```bash
# From harness/
uv run python score_local.py --model vals-hackathon/base --balanced-play-ablation \
  --games 24 --lineup mixed --out runs/balanced-pilot
uv run python score_local.py --model vals-hackathon/base --balanced-play-ablation \
  --games 24 --seed 960510 --lineup house --out runs/balanced-holdout
```

Collect source traces on separate training deals. Keep seeds and complete source games out of the pilot and
holdout sets. Use a dedicated trace-source deployment with the gateway builder to keep evaluation calls out of
the training dataset. One-step OPSD samples a fresh student action given each saved context.

```bash
# From the repository root. Both commands preview by default or with --dry-run.
uv run python opsd/build_from_gateway.py --deployment-id <train-infer-id> \
  --dataset pokerbots-balanced-one-step --strategy balanced --rollout-mode one_step \
  --opponent-reads --table-talk-defense
uv run python opsd/train.py --dataset pokerbots-balanced-one-step --rollout-mode one_step --dry-run
```

Review the builder's `preflop_reference_kept` count to verify coverage of the blind-defense comparison.
After reviewing the dataset, rerun the builder with `--upload` and the trainer without `--dry-run`. Compare
each candidate checkpoint under the official prompt to both earlier deployments on held-out full games.

```bash
# From harness/
uv run python score_local.py --model vals-hackathon/candidate --compare-model vals-hackathon/base \
  --games 48 --seed 173050 --lineup mixed --out runs/candidate-vs-base
```

Use a fresh seed block after selecting the checkpoint. Compare paired placement, busts, failed actions, and
folds in the flagged spots. Check the counts and review individual decisions before attributing a fold-rate
change to better play.

## Run

1. **Play base on train deals with the official event prompt.** For example, use a window of a 30-hand
   game from a house dataset built by `rl/make_training_datasets.py`:
   ```bash
   uv run python -m vals_tools.eval --dataset pokerbots-reads-v1-house-train --deployment <project>/<base-deployment>
   ```
2. **Build the hinted dataset** from that eval's traces ([`build_dataset.py`](build_dataset.py)):
   ```bash
   uv run python opsd/build_dataset.py --eval <eval-id> --dataset pokerbots-opsd-train --dry-run
   uv run python opsd/build_dataset.py --eval <eval-id> --dataset pokerbots-opsd-train
   ```
3. **Train** ([`train.py`](train.py)): use early checkpoints to compare the effect of distillation. The learning rate and
   lengths are AC2's defaults unless you pass `--lr`, `--max-response-len` or `--max-total-len`; `--dry-run` lists
   every setting the launch sends.
   ```bash
   uv run python opsd/train.py --dataset pokerbots-opsd-train --checkpoint-every 2 --dry-run
   uv run python opsd/train.py --dataset pokerbots-opsd-train --checkpoint-every 2
   ```
4. **Evaluate the student without the hint:** serve a checkpoint and run it through the harness on
   `pokerbots-eval`. The harness never shows the hint, so this measures what the weights learned.

## The `HintedTask` contract

There is no trace-to-`HintedTask` helper, so `build_dataset.py` builds the tasks itself. Each policy decision
becomes one task:

- `input`: the decision's prompt. Offline mode also stores the reply base wrote (its reasoning, text and
  `submit_action` call);
- `opsd_hint_selections`: one `HintSelection` with `rollout_prefix_len` at the end of the prompt and
  `HintMutation(item_boundary=1, action="append_to_previous_item")`, which appends the hint to the system message
  for the teacher only;
- `env_params.tools`: the tool schemas the model saw, `submit_action` and `update_scratchpad` ([`tools.json`](tools.json),
  generated from the harness and checked by a harness test). **Training rejects a transcript with tool calls unless
  its tools are listed here.**

The dataset is created with `required_types=["hinted"]`. Only clean decisions are kept: one `submit_action` call
and a reply under the token cap. The after-hand `update_scratchpad` turns are skipped.

## What offline mode can and can't do

- It replays base's own tokens, so it can only reweight behavior base already produces. A hint asking for something
  base never writes needs `rollout_mode="one_step"`, where the student samples its own reply.
- It is cheap: no inference engines and no opponent calls during training. Opponent calls happen only in steps 1
  and 4.

## Judge it

Compare the student with base on `pokerbots-eval`, both served the same way, paired per deal, with
`python -m vals_tools.score`.

## Learn opponent reads

Test the hint on the existing deployment before using a training slot. Both arms get the same placement guidance.
The opponent-read arm additionally gets guidance for recording observations in cleanup and applying them on later
decisions. Each pair uses the same deck seed, starting seat and opponent lineup. The scorer reports paired placement
and saves each arm's replay in its own directory.

```bash
cd harness
uv run python score_local.py --model vals-hackathon/entry --opponent-read-ablation --games 8 --out runs/reads-pilot
uv run python score_local.py --model vals-hackathon/entry --opponent-read-ablation --games 8 \
  --seed 960510 --talk-mode shuffled --lineup mixed --out runs/reads-holdout
```

Inspect which bets change and whether the notes reflect the visible history. Use `--talk-mode muted` to remove
scripted chatter while retaining the event prompt and tool schemas. Run the official prompt on the same seeds as
a further baseline by omitting `--opponent-read-ablation`.

If the hint improves placement, use multi-hand training deals for the source eval. The longer tasks let the
base model write a note and then play a hand with it. The cleanup hint is applied to valid scratchpad updates.
The action hint is applied to poker decisions. Check the `cleanup_kept` count before launching training.

```bash
# From the repository root. Omit --upload for a dry run.
uv run python rl/make_training_datasets.py --family house --prefix pokerbots-reads-v1 \
  --house-hands 3 --house-talk shuffled --count 240 --upload
uv run python -m vals_tools.eval --dataset pokerbots-reads-v1-house-train --deployment vals-hackathon/entry
uv run python opsd/build_dataset.py --eval <eval-id> --dataset pokerbots-opsd-reads-v1 --opponent-reads --dry-run
```

When the dataset checks pass, rerun the builder without `--dry-run` and use the standard offline training command.
Compare the checkpoint on fresh full-game seeds with the official prompt. Keep all decisions and cleanup turns
from a source game in the same split. Replay identities and unrevealed hole cards can inform offline analysis and
should stay out of model inputs and hints.

## Build from gateway traces (no eval needed)

Every request a deployment answers is recorded at the inference gateway with its prompt and reply, whoever sent it.
[`build_from_gateway.py`](build_from_gateway.py) builds the same hinted dataset from those records, so decisions the
base deployment made in arena games and in local games (`harness/play_local.py`, `harness/score_local.py`) can be
used without running an AC2 eval through the harness relay.

```bash
uv run python opsd/build_from_gateway.py --deployment-id <infer_...> --dataset pokerbots-opsd-tight            # dry run
uv run python opsd/build_from_gateway.py --deployment-id <infer_...> --dataset pokerbots-opsd-tight --upload
```

## Table talk defense

Use the paired test to compare the official prompt, guidance to ignore chatter, and guidance that gives specific
claims some weight. With `--talk-mode false-claims`, the scripted house seats sometimes name a pocket pair they
do not hold when they raise preflop. The arms share seeds and lineups. The scorer reports the paired difference
between ignoring chatter and using specific claims, in rating points per game.

```bash
# From harness/
uv run python score_local.py --model vals-hackathon/entry --table-talk-defense-ablation \
  --talk-mode false-claims --lineup house --seed 960510 --out runs/talk-defense
```

On a table you host locally, `play_local.py --ignore-table-talk` removes quoted opponent messages from the
evaluated seat's action prompts and recaps. Cards, betting actions, and tool schemas stay available to the seat.
The arena hosts its own table, so evaluate the trained checkpoint with the official prompt and chatter present.

```bash
# From the repository root. The builder prints counts by default.
uv run python opsd/build_from_gateway.py --deployment-id <infer_...> \
  --dataset pokerbots-opsd-talk-defense --table-talk-defense
```

Use base-model traces from training games. The action hint teaches the model to ignore chatter when deciding.
The cleanup hint teaches it to keep player prose out of the scratchpad. Check `cleanup_kept` and review a sample
of both turn types before uploading. Use `--opponent-reads` as well if the experiment calls for both forms of guidance.

Both builders require the unmodified event system prompt. Preview and treatment requests with guidance appended
are skipped. `--hint-file` replaces the teacher strategy and can be combined with `--opponent-reads` and
`--table-talk-defense`. Use the training base deployment for offline mode, which replays its own response tokens.
`--since <UTC time>` limits the gateway source to recent requests.

To inspect the configuration for a one-step smoke run, use a separate dataset built with
`--rollout-mode one_step` and run:

```bash
uv run python opsd/train.py --dataset pokerbots-balanced-one-step --rollout-mode one_step \
  --steps 1 --batch 8 --checkpoint-every 1 --dry-run
```

The preview checks the local configuration. A scheduled smoke launch is needed to verify that the org accepts it.

## First-in open card

The open card estimates the rating value of a small preflop open and a fold when every earlier player has folded.
It uses the seat's hole cards, position relative to the button, players remaining and effective stack. The standard
raise target is 2.5 big blinds. Historical raises of 2–3 big blinds supply the estimate. Limp and larger-raise
outcomes are retained in the report. Sparse cells pool related observations. The card marks small or unsupported
edges as unclear for teacher collection.

Run from the repository root with a synced harness environment.

```bash
python3 arena_tools/fetch_replays.py arena_data
(cd harness && uv run python price_spots.py --kind open --replays ../arena_data)
uv run --project harness python opsd/build_price_tables.py --replays arena_data \
  --open-spots harness/runs/open_spots.jsonl --open-report harness/runs/open-heldout.json
uv run --project harness python opsd/simulate_cards.py --games 256 --seed 20272000
```

Both the stack curves and the open card are fit on training matches. The report measures stack value after each
hand minus stack value before the blinds, with the recorded place used for a bust. It compares the policies on
held-out cells with at least three observed folds and three standard raises. Archived actions reflect the choices
of different opponents and give one observed outcome per decision. The CPU simulation tests the card on a shared
scripted policy across house, mixed and tight tables.

To collect opening traces from an existing teacher deployment, add
`--decision open --spots harness/runs/open_spots.jsonl` to both `opsd/price_run.py` and
`opsd/build_price_dataset.py`. Use `--hint --balance` on the collection step. The builder leaves out held-out
spots and checks that a hinted reply follows the card with a legal small raise or fold.

## Terminal call teacher

`build_terminal_teacher.py` checks heads-up all-ins and river calls that close the action against the arena replay
payouts. The teacher settles both choices, including dead chips, splits and uncalled chips, and scores the resulting
stack by expected finishing place. It estimates the opponent's range using betting frequencies from training matches
with a prior fitted to Pluribus's published decisions. The report compares the same procedure with field data alone.
The teacher exports only training decisions where the call or fold is favored across its three ranges.

```bash
uv run --project harness python opsd/build_terminal_teacher.py --replays arena_data \
  --pluribus /path/to/phh-dataset/data/pluribus --spots harness/runs/arena-call-spots.jsonl \
  --fresh harness/runs/post-open-card-call-spots.jsonl --out harness/runs/terminal-teacher --samples 512
uv run --project harness python opsd/simulate_cards.py \
  --terminal-ranges harness/runs/terminal-teacher/ranges.json --seed 20288000 --games 512
```

The fresh call bank should come from matches disjoint from the original bank. The local SFT dry run uses
`uv run python sft/build_dataset.py --examples harness/runs/terminal-teacher/examples.jsonl --dataset pokerbots-terminal-calls --max-fold-share 0.6`.
For an AC2 run preview, use `uv run python sft/train.py --dataset pokerbots-terminal-calls --from-checkpoint <parent> --steps 4 --batch 32 --checkpoint-every 2 --dry-run`.
