"""Gymnasium environment implementing Robots Evolved movement rules."""

from __future__ import annotations

from typing import Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from robotsevolved_env.generator import generate_puzzle
from robotsevolved_env.puzzle import Puzzle

DIRECTIONS: tuple[tuple[int, int], ...] = ((-1, 0), (0, 1), (1, 0), (0, -1))
DIRECTION_NAMES = ("UP", "RIGHT", "DOWN", "LEFT")


class RobotsEvolvedEnv(gym.Env[dict[str, np.ndarray], int]):
    """One action selects a robot and direction; the robot slides until blocked."""

    metadata = {"render_modes": ["ansi", "rgb_array"], "render_fps": 8}

    def __init__(
        self,
        puzzle: Puzzle | None = None,
        *,
        width: int = 12,
        height: int = 12,
        wall_probability: float = 0.1,
        max_steps: int = 256,
        reward_mode: Literal["sparse", "dense"] = "dense",
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.fixed_puzzle = puzzle
        self.width = puzzle.width if puzzle else width
        self.height = puzzle.height if puzzle else height
        self.wall_probability = wall_probability
        self.max_steps = max_steps
        self.reward_mode = reward_mode
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(16)
        self.observation_space = spaces.Dict({
            "robots": spaces.Box(0, max(self.width, self.height) - 1, (4, 2), np.int16),
            "goal": spaces.Box(0, max(self.width, self.height) - 1, (2,), np.int16),
            "target_robot": spaces.Box(-1, 3, (1,), np.int8),
            "horizontal_walls": spaces.MultiBinary((self.height + 1, self.width)),
            "vertical_walls": spaces.MultiBinary((self.height, self.width + 1)),
            "action_mask": spaces.MultiBinary(16),
        })
        self.puzzle: Puzzle
        self.robots: list[tuple[int, int]] = []
        self.steps = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        del options
        self.puzzle = self.fixed_puzzle or generate_puzzle(
            self.width, self.height, self.wall_probability, rng=self.np_random
        )
        self.robots = list(self.puzzle.robots)
        self.steps = 0
        return self._observation(), {"action_mask": self.action_masks()}

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Action must be 0..15, got {action!r}")
        robot, direction = divmod(int(action), 4)
        previous_distance = self._goal_distance()
        destination = self._destination(robot, direction)
        moved = destination != self.robots[robot]
        if moved:
            self.robots[robot] = destination
        self.steps += 1
        terminated = self._is_solved()
        truncated = self.steps >= self.max_steps and not terminated

        if terminated:
            reward = 10.0
        elif not moved:
            reward = -1.0
        elif self.reward_mode == "dense":
            reward = -0.05 + 0.1 * (previous_distance - self._goal_distance())
        else:
            reward = -0.01
        mask = self.action_masks()
        info = {"moved": moved, "steps": self.steps, "action_mask": mask}
        return self._observation(mask), reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        return np.fromiter(
            (self._destination(robot, direction) != self.robots[robot] for robot in range(4) for direction in range(4)),
            dtype=np.int8,
            count=16,
        )

    def _destination(self, robot: int, direction: int) -> tuple[int, int]:
        row, col = self.robots[robot]
        dr, dc = DIRECTIONS[direction]
        occupied = set(self.robots)
        occupied.remove((row, col))
        while True:
            next_row, next_col = row + dr, col + dc
            if (next_row, next_col) in occupied or self._wall_blocks(row, col, direction):
                return row, col
            row, col = next_row, next_col

    def _wall_blocks(self, row: int, col: int, direction: int) -> bool:
        if direction == 0:
            return (row, col) in self.puzzle.horizontal_walls
        if direction == 1:
            return (row, col + 1) in self.puzzle.vertical_walls
        if direction == 2:
            return (row + 1, col) in self.puzzle.horizontal_walls
        return (row, col) in self.puzzle.vertical_walls

    def _is_solved(self) -> bool:
        if self.puzzle.target_robot is None:
            return self.puzzle.goal in self.robots
        return self.robots[self.puzzle.target_robot] == self.puzzle.goal

    def _goal_distance(self) -> int:
        candidates = self.robots if self.puzzle.target_robot is None else [self.robots[self.puzzle.target_robot]]
        return min(abs(row - self.puzzle.goal[0]) + abs(col - self.puzzle.goal[1]) for row, col in candidates)

    def _observation(self, mask: np.ndarray | None = None) -> dict[str, np.ndarray]:
        horizontal = np.zeros((self.height + 1, self.width), dtype=np.int8)
        vertical = np.zeros((self.height, self.width + 1), dtype=np.int8)
        for row, col in self.puzzle.horizontal_walls:
            if 0 <= row <= self.height and 0 <= col < self.width:
                horizontal[row, col] = 1
        for row, col in self.puzzle.vertical_walls:
            if 0 <= row < self.height and 0 <= col <= self.width:
                vertical[row, col] = 1
        return {
            "robots": np.asarray(self.robots, dtype=np.int16),
            "goal": np.asarray(self.puzzle.goal, dtype=np.int16),
            "target_robot": np.asarray([-1 if self.puzzle.target_robot is None else self.puzzle.target_robot], dtype=np.int8),
            "horizontal_walls": horizontal,
            "vertical_walls": vertical,
            "action_mask": self.action_masks() if mask is None else mask,
        }

    def render(self) -> str | np.ndarray | None:
        if self.render_mode == "rgb_array":
            return self._render_rgb()
        grid = [["." for _ in range(self.width)] for _ in range(self.height)]
        goal_row, goal_col = self.puzzle.goal
        grid[goal_row][goal_col] = "*"
        for index, (row, col) in enumerate(self.robots):
            grid[row][col] = "BGRY"[index]
        text = "\n".join(" ".join(row) for row in grid)
        if self.render_mode == "ansi":
            return text
        print(text)
        return None

    def _render_rgb(self) -> np.ndarray:
        cell = 24
        image = np.full((self.height * cell, self.width * cell, 3), 235, dtype=np.uint8)
        colors = ((65, 105, 225), (34, 139, 34), (178, 34, 34), (255, 140, 0))
        row, col = self.puzzle.goal
        image[row * cell + 8 : row * cell + 16, col * cell + 8 : col * cell + 16] = (25, 25, 25)
        for index, (row, col) in enumerate(self.robots):
            image[row * cell + 4 : row * cell + 20, col * cell + 4 : col * cell + 20] = colors[index]
        for row, col in self.puzzle.horizontal_walls:
            if 0 <= row <= self.height:
                y = min(row * cell, image.shape[0] - 2)
                image[y : y + 2, col * cell : (col + 1) * cell] = 0
        for row, col in self.puzzle.vertical_walls:
            if 0 <= col <= self.width:
                x = min(col * cell, image.shape[1] - 2)
                image[row * cell : (row + 1) * cell, x : x + 2] = 0
        return image
