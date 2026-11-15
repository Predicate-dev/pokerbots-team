# Offline OPSD: distill a strategy hint

At the event the harness owns the system prompt, so any strategy you want the model to follow has to be in its
weights. Offline OPSD (on-policy self-distillation) puts it there: the teacher is the base model with a hint in its
system prompt, the student is the same model without it, and training pulls the student's token distribution
toward the teacher's on replies the base model actually wrote.

[`hint.py`](hint.py) holds a deliberately generic tight-aggressive hint. Writing a better one is the point.

## Run

1. **Play base on the train deals** through the harness, served as an inference deployment:
   ```bash
   uv run python -m vals_tools.eval --dataset pokerbots-train --deployment <project>/<base-deployment>
   ```
2. **Build the hinted dataset** from that eval's traces ([`build_dataset.py`](build_dataset.py)):
   ```bash
   uv run python opsd/build_dataset.py --eval <eval-id> --dataset pokerbots-opsd-train --dry-run
   uv run python opsd/build_dataset.py --eval <eval-id> --dataset pokerbots-opsd-train
   ```
3. **Train** ([`train.py`](train.py)): 20 steps of 64 decisions, a checkpoint every 10. The learning rate and
   lengths are AC2's defaults unless you pass `--lr`, `--max-response-len` or `--max-total-len`; `--dry-run` lists
   every setting the launch sends.
   ```bash
   uv run python opsd/train.py --dataset pokerbots-opsd-train --dry-run
   uv run python opsd/train.py --dataset pokerbots-opsd-train
   ```
4. **Evaluate the student without the hint:** serve a checkpoint and run it through the harness on
   `pokerbots-eval`. The harness never shows the hint, so this measures what the weights learned.

## The `HintedTask` contract

There is no trace-to-`HintedTask` helper, so `build_dataset.py` builds the tasks itself. Each policy decision
becomes one task:

- `input`: the decision's prompt followed by the reply base wrote (its reasoning, text and `submit_action` call);
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

Use the paired test to compare the official prompt with the same prompt plus guidance to ignore other players'
chatter. Both arms see the same scripted lineup and types of chatter. Try fresh seeds and a mixed lineup with
shuffled chatter before using the result to choose a training run. Also check a quiet lineup for loss of placement
or reliability.

```bash
# From harness/
uv run python score_local.py --model vals-hackathon/entry --table-talk-defense-ablation \
  --talk-mode shuffled --lineup mixed --seed 960510 --out runs/talk-defense
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

It keeps the same decisions as `build_dataset.py` (one `submit_action` call, under the token cap) and skips
scratchpad turns and non-poker requests. Use the base deployment's id only: offline OPSD replays the base model's
own tokens. `--since <UTC time>` limits it to recent requests.
