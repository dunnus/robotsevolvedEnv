"""Observation encodings for tensor policies and language-model agents."""

from __future__ import annotations

from robotsevolved_env.puzzle import Puzzle, ROBOT_COLORS

ROBOT_LETTERS = "BGRY"


def text_observation(puzzle: Puzzle) -> str:
    """Encode the complete board as compact, deterministic text for an LLM."""
    robots = ", ".join(
        f"{letter}=({row},{col})"
        for letter, (row, col) in zip(ROBOT_LETTERS, puzzle.robots)
    )
    horizontal = " ".join(f"({row},{col})" for row, col in sorted(puzzle.horizontal_walls))
    vertical = " ".join(f"({row},{col})" for row, col in sorted(puzzle.vertical_walls))
    target = "any robot" if puzzle.target_robot is None else ROBOT_COLORS[puzzle.target_robot]
    return (
        f"Board: {puzzle.height} rows x {puzzle.width} columns.\n"
        f"Robots (row,column): {robots}.\n"
        f"Goal: ({puzzle.goal[0]},{puzzle.goal[1]}), target={target}.\n"
        "A horizontal wall (r,c) blocks crossing between rows r-1 and r in column c.\n"
        f"Horizontal walls: {horizontal}.\n"
        "A vertical wall (r,c) blocks crossing between columns c-1 and c in row r.\n"
        f"Vertical walls: {vertical}."
    )


def training_prompt(puzzle: Puzzle, max_moves: int = 20) -> str:
    """Build the one-shot prompt used by TRL rollouts."""
    return (
        "Solve this sliding-robot puzzle. A selected robot moves N/E/S/W until a wall or "
        "another robot stops it. Return only a JSON array of moves such as "
        f"[\"GW\",\"BS\"], using at most {max_moves} moves.\n\n"
        + text_observation(puzzle)
    )