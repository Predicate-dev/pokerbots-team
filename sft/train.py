"""Supervised fine-tuning (SFT) on stored conversations, e.g. another entry's actions rebuilt from arena replays.

AC2's `sft` method trains directly on each task's `input` (the full prompt and reply). It runs no environment, no
grader and no inference engine, so like offline OPSD it never touches the harness relay. Training has no poker
eval: afterwards, deploy a checkpoint and score it with `harness/score_local.py`.

    uv run python sft/train.py --dataset pokerbots-sft-pete --dry-run
    uv run python sft/train.py --dataset pokerbots-sft-pete
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from typing import Any

from ac2.sdk import Client
from ac2.sdk.model import ModelConfig
from ac2.sdk.run_config import (
    EvalConfig,
    ExperimentConfig,
    RolloutConfig,
    RunConfig,
    SFTConfig,
    StepConfig,
    TasksConfig,
    TrainDatasetConfig,
    TrainerConfig,
)


def _set(**values: Any) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def run_config(args: argparse.Namespace, name: str) -> RunConfig:
    return RunConfig(
        # `--from-checkpoint` continues one of our own checkpoints (e.g. an OPSD checkpoint) instead of the base model.
        model=ModelConfig(checkpoint=args.from_checkpoint) if args.from_checkpoint else args.model,
        name=name,
        description=f"SFT of {args.from_checkpoint or args.model} on the conversations in {args.dataset}.",
        tags={"benchmark": "pokerbots", "method": "sft", "dataset": args.dataset},
        experiment_cfg=ExperimentConfig(
            method=SFTConfig(),
            step=StepConfig(
                samples_per_problem=1,
                **_set(num_train_steps=args.steps, problem_batch_size=args.batch,
                       checkpoint_interval=args.checkpoint_every, keep_last_checkpoints=args.keep_checkpoints),
            ),
            # AC2 rejects `apply_chat_template_kwargs` for sft ("sft training does not use it"), so none is sent.
            tasks=TasksConfig(datasets=(TrainDatasetConfig(ac2_dataset_name=args.dataset),)),
            trainer=TrainerConfig(**_set(learning_rate=args.lr)),
            rollout=RolloutConfig(n_inference_replicas=0, **_set(max_total_len=args.max_total_len)),
            eval=EvalConfig(mode="off"),
        ),
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", default="vals-hackathon")
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    p.add_argument("--from-checkpoint", help="start from one of our checkpoints (ckpt_...) instead of the base model")
    p.add_argument("--steps", type=int, default=15)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--checkpoint-every", type=int, default=5)
    p.add_argument("--lr", type=float)
    p.add_argument("--max-total-len", type=int)
    p.add_argument("--keep-checkpoints", type=int)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    name = f"pokerbots-sft-{datetime.now(UTC):%Y%m%d-%H%M%S}"
    config = run_config(args, name)
    if args.dry_run:
        print(json.dumps(config.to_json(), indent=1, default=str))
        return
    run = Client(project=args.project).train.run(config)
    print(json.dumps({"train_id": run.train_id, "name": name}))


if __name__ == "__main__":
    main()
