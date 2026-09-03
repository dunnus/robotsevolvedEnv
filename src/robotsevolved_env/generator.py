"""Deterministic offline puzzle generation."""

from __future__ import annotations

import numpy as np

from robotsevolved_env.puzzle import Puzzle


def generate_puzzle(
    width: int = 12,
    height: int = 12,
    wall_probability: float = 0.1,
    *,
    rng: np.random.Generator | None = None,
) -> Puzzle:
    """Generate a board for training; solvability is not guaranteed."""
    if not 0 <= wall_probability <= 1:
        raise ValueError("wall_probability must be between 0 and 1")
    rng = rng or np.random.default_rng()
    cells = rng.choice(width * height, size=5, replace=False)
    positions = tuple((int(cell // width), int(cell % width)) for cell in cells)
    horizontal = {(row, col) for col in range(width) for row in (0, height)}
    vertical = {(row, col) for row in range(height) for col in (0, width)}
    for row in range(1, height):
        for col in range(width):
            if rng.random() < wall_probability:
                horizontal.add((row, col))
    for row in range(height):
        for col in range(1, width):
            if rng.random() < wall_probability:
                vertical.add((row, col))
    return Puzzle(
        width=width,
        height=height,
        robots=positions[:4],  # type: ignore[arg-type]
        horizontal_walls=frozenset(horizontal),
        vertical_walls=frozenset(vertical),
        goal=positions[4],
    )
