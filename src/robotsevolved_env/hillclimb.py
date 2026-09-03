"""Mutation hill-climber baseline for action sequences."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from robotsevolved_env.env import RobotsEvolvedEnv
from robotsevolved_env.generator import generate_puzzle
from robotsevolved_env.puzzle import Puzzle


@dataclass(frozen=True)
class Candidate:
    actions: np.ndarray
    score: float
    solved: bool
    steps: int


def evaluate(puzzle: Puzzle, actions: np.ndarray) -> Candidate:
    env = RobotsEvolvedEnv(puzzle, reward_mode="dense", max_steps=len(actions))
    env.reset()
    total = 0.0
    solved = False
    steps = 0
    for action in actions:
        _, reward, solved, truncated, _ = env.step(int(action))
        total += reward
        steps += 1
        if solved or truncated:
            break
    if solved:
        total += 1000.0 - steps
    return Candidate(actions.copy(), total, solved, steps)


def hill_climb(puzzle: Puzzle, *, length: int = 64, iterations: int = 10_000, seed: int = 0) -> Candidate:
    rng = np.random.default_rng(seed)
    best = evaluate(puzzle, rng.integers(0, 16, size=length, dtype=np.int16))
    for _ in range(iterations):
        proposal = best.actions.copy()
        changes = int(rng.integers(1, max(2, length // 8)))
        indices = rng.choice(length, size=changes, replace=False)
        proposal[indices] = rng.integers(0, 16, size=changes)
        candidate = evaluate(puzzle, proposal)
        if candidate.score >= best.score:
            best = candidate
        if best.solved:
            best = Candidate(best.actions[: best.steps], best.score, True, best.steps)
            break
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    puzzle = generate_puzzle(rng=np.random.default_rng(args.seed))
    result = hill_climb(puzzle, length=args.length, iterations=args.iterations, seed=args.seed)
    print(f"solved={result.solved} score={result.score:.2f} steps={result.steps}")
    print("actions:", " ".join(map(str, result.actions.tolist())))


if __name__ == "__main__":
    main()
