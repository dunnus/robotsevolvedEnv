"""Replay and score action-sequence rollouts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Sequence

from robotsevolved_env.env import RobotsEvolvedEnv
from robotsevolved_env.puzzle import Puzzle, Position

ROBOT_INDEX = {letter: index for index, letter in enumerate("BGRY")}
DIRECTION_INDEX = {letter: index for index, letter in enumerate("NESW")}


@dataclass(frozen=True, slots=True)
class ReplayResult:
    """State trajectory and aggregate result for one proposed move sequence."""

    moves: tuple[str, ...]
    states: tuple[tuple[Position, Position, Position, Position], ...]
    rewards: tuple[float, ...]
    moved: tuple[bool, ...]
    solved: bool
    invalid_moves: int


def move_to_action(move: str) -> int:
    """Convert a move such as ``BE`` to the Gymnasium discrete action."""
    token = move.strip().upper()
    if len(token) != 2 or token[0] not in ROBOT_INDEX or token[1] not in DIRECTION_INDEX:
        raise ValueError(f"Invalid move {move!r}; expected B/G/R/Y followed by N/E/S/W")
    return ROBOT_INDEX[token[0]] * 4 + DIRECTION_INDEX[token[1]]


def parse_completion(completion: object) -> list[str]:
    """Extract a JSON move array from a plain or conversational TRL completion."""
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        completion = completion[-1].get("content", "")
    if not isinstance(completion, str):
        raise ValueError("Completion is not text")
    match = re.search(r"\[[\s\S]*?\]", completion)
    if match is None:
        raise ValueError("Completion contains no JSON array")
    moves = json.loads(match.group(0))
    if not isinstance(moves, list) or not all(isinstance(move, str) for move in moves):
        raise ValueError("Completion must contain a JSON array of move strings")
    return [move.strip().upper() for move in moves]


def replay_moves(puzzle: Puzzle, moves: Sequence[str], *, max_steps: int | None = None) -> ReplayResult:
    """Replay moves locally, retaining every robot configuration for visualization."""
    limit = max_steps if max_steps is not None else max(1, len(moves))
    env = RobotsEvolvedEnv(puzzle, reward_mode="dense", max_steps=limit)
    env.reset()
    states: list[tuple[Position, Position, Position, Position]] = [puzzle.robots]
    rewards: list[float] = []
    moved: list[bool] = []
    accepted: list[str] = []
    invalid = 0
    solved = False
    for raw_move in moves[:limit]:
        try:
            action = move_to_action(raw_move)
        except ValueError:
            invalid += 1
            continue
        _, reward, solved, truncated, info = env.step(action)
        accepted.append(raw_move.strip().upper())
        rewards.append(float(reward))
        moved.append(bool(info["moved"]))
        states.append(tuple(env.robots))  # type: ignore[arg-type]
        if solved or truncated:
            break
    return ReplayResult(
        moves=tuple(accepted),
        states=tuple(states),
        rewards=tuple(rewards),
        moved=tuple(moved),
        solved=solved,
        invalid_moves=invalid,
    )


def sequence_reward(puzzle: Puzzle, moves: Sequence[str], *, max_moves: int = 20) -> float:
    """Return a bounded reward suitable for language-model policy optimization."""
    result = replay_moves(puzzle, moves, max_steps=max_moves)
    no_ops = sum(not moved for moved in result.moved)
    if result.solved:
        efficiency = max(0, max_moves - len(result.moves)) / max_moves
        return min(1.0, 0.75 + 0.25 * efficiency - 0.02 * no_ops)
    goal_row, goal_col = puzzle.goal
    candidates = result.states[-1]
    if puzzle.target_robot is not None:
        candidates = (candidates[puzzle.target_robot],)  # type: ignore[assignment]
    distance = min(abs(row - goal_row) + abs(col - goal_col) for row, col in candidates)
    span = max(1, puzzle.width + puzzle.height - 2)
    return max(0.0, 0.35 * (1.0 - distance / span) - 0.02 * (result.invalid_moves + no_ops))