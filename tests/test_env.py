from __future__ import annotations

import numpy as np

from robotsevolved_env.env import RobotsEvolvedEnv
from robotsevolved_env.generator import generate_puzzle
from robotsevolved_env.puzzle import Puzzle
from robotsevolved_env.rollout import parse_completion, replay_moves, sequence_reward


def board(**changes: object) -> Puzzle:
    values = dict(
        width=5,
        height=5,
        robots=((2, 1), (2, 3), (0, 0), (4, 4)),
        horizontal_walls=frozenset((row, col) for col in range(5) for row in (0, 5)),
        vertical_walls=frozenset((row, col) for row in range(5) for col in (0, 5)),
        goal=(2, 2),
    )
    values.update(changes)
    return Puzzle(**values)  # type: ignore[arg-type]


def test_robot_stops_before_another_robot_and_wins() -> None:
    env = RobotsEvolvedEnv(board(), max_steps=10)
    env.reset()
    _, reward, terminated, truncated, info = env.step(1)  # blue right
    assert env.robots[0] == (2, 2)
    assert terminated and not truncated and reward == 10
    assert info["moved"] is True


def test_internal_wall_stops_robot() -> None:
    puzzle = board(vertical_walls=board().vertical_walls | {(2, 2)}, goal=(4, 3))
    env = RobotsEvolvedEnv(puzzle)
    env.reset()
    env.step(1)
    assert env.robots[0] == (2, 1)


def test_invalid_noop_is_masked_and_penalized() -> None:
    puzzle = board(robots=((0, 0), (2, 3), (3, 2), (4, 4)))
    env = RobotsEvolvedEnv(puzzle)
    _, info = env.reset()
    assert info["action_mask"][0] == 0
    _, reward, terminated, truncated, step_info = env.step(0)
    assert reward == -1 and not terminated and not truncated
    assert step_info["moved"] is False


def test_truncation() -> None:
    env = RobotsEvolvedEnv(board(goal=(4, 0)), max_steps=1)
    env.reset()
    _, _, terminated, truncated, _ = env.step(0)
    assert not terminated and truncated


def test_colored_goal_requires_target_robot() -> None:
    puzzle = board(goal=(2, 2), target_robot=1)
    env = RobotsEvolvedEnv(puzzle)
    env.reset()
    _, _, terminated, _, _ = env.step(1)
    assert not terminated


def test_reset_seed_is_deterministic() -> None:
    env = RobotsEvolvedEnv(width=7, height=7)
    first, _ = env.reset(seed=42)
    second, _ = env.reset(seed=42)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])


def test_generator_seed_is_deterministic() -> None:
    first = generate_puzzle(rng=np.random.default_rng(7))
    second = generate_puzzle(rng=np.random.default_rng(7))
    assert first == second


def test_action_mask_matches_destinations() -> None:
    env = RobotsEvolvedEnv(board())
    env.reset()
    mask = env.action_masks()
    assert mask.shape == (16,)
    assert mask[1] == 1


def test_replay_and_completion_parser() -> None:
    moves = parse_completion('answer: ```json\n["BE"]\n```')
    result = replay_moves(board(), moves)
    assert moves == ["BE"]
    assert result.solved
    assert result.states[-1][0] == (2, 2)
    assert sequence_reward(board(), moves) > 0.75
