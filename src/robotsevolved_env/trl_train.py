"""Offline GRPO training for language models using local puzzle snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from robotsevolved_env.data import load_records
from robotsevolved_env.observation import training_prompt
from robotsevolved_env.puzzle import Puzzle
from robotsevolved_env.rollout import parse_completion, sequence_reward


def build_training_rows(data_dir: str | Path, *, repeats: int = 128, max_moves: int = 20) -> list[dict[str, Any]]:
    """Convert local puzzle records into standard-format TRL rows."""
    rows: list[dict[str, Any]] = []
    records = load_records(data_dir)
    for _ in range(repeats):
        for record in records:
            puzzle = Puzzle.from_json(record["puzzle"])
            rows.append({
                "prompt": training_prompt(puzzle, max_moves),
                "puzzle_json": json.dumps(record["puzzle"], separators=(",", ":")),
                "max_moves": max_moves,
            })
    return rows


def puzzle_reward(
    completions: Sequence[object],
    puzzle_json: Sequence[str],
    max_moves: Sequence[int],
    **_: Any,
) -> list[float]:
    """Score TRL completions by parsing and replaying moves in the local simulator."""
    rewards: list[float] = []
    for completion, encoded, limit in zip(completions, puzzle_json, max_moves):
        try:
            puzzle = Puzzle.from_json(json.loads(encoded))
            moves = parse_completion(completion)
            rewards.append(sequence_reward(puzzle, moves, max_moves=int(limit)))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            rewards.append(0.0)
    return rewards


def format_reward(completions: Sequence[object], **_: Any) -> list[float]:
    """Reward machine-readable JSON move arrays."""
    rewards = []
    for completion in completions:
        try:
            parse_completion(completion)
            rewards.append(0.1)
        except (TypeError, ValueError, json.JSONDecodeError):
            rewards.append(0.0)
    return rewards


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to a locally available instruct model")
    parser.add_argument("--data-dir", type=Path, default=Path("data/puzzles"))
    parser.add_argument("--output-dir", default="artifacts/trl-grpo")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=128)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-completion-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        parser.error("--model must be an existing local model path")
    if args.batch_size % args.num_generations != 0:
        parser.error("--batch-size must be divisible by --num-generations")
    if args.steps < 1 or args.repeats < 1:
        parser.error("--steps and --repeats must be positive")

    try:
        import torch
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as error:
        raise SystemExit('Install training dependencies with: pip install -e ".[train]"') from error

    dataset = Dataset.from_list(build_training_rows(args.data_dir, repeats=args.repeats))
    config = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        remove_unused_columns=False,
        scale_rewards=False,
        loss_type="dr_grpo",
        mask_truncated_completions=True,
        log_completions=True,
        report_to="none",
        use_cpu=args.cpu,
        bf16=not args.cpu and torch.cuda.is_bf16_supported(),
        model_init_kwargs={"local_files_only": True},
    )
    trainer = GRPOTrainer(
        model=args.model,
        args=config,
        reward_funcs=[puzzle_reward, format_reward],
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()