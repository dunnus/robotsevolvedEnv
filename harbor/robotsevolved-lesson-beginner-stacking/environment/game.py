#!/usr/bin/env python3
"""Offline CLI and verifier engine for one sliding-robot lesson."""
from __future__ import annotations
import json
import sys
from pathlib import Path

COLORS = "BGRY"
DIRECTIONS = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1)}
PUZZLE_PATH = Path("/data/puzzle.json")

def load(path: Path = PUZZLE_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def move(puzzle: dict, robots: dict[str, list[int]], token: str) -> bool:
    token = token.upper()
    if len(token) != 2 or token[0] not in COLORS or token[1] not in DIRECTIONS:
        raise ValueError(f"invalid move {token!r}; expected B/G/R/Y plus N/E/S/W")
    robot, direction = token
    row, col = robots[robot]
    dr, dc = DIRECTIONS[direction]
    occupied = {tuple(position) for name, position in robots.items() if name != robot}
    horizontal = {tuple(wall) for wall in puzzle["horizontal_walls"]}
    vertical = {tuple(wall) for wall in puzzle["vertical_walls"]}
    start = (row, col)
    while True:
        blocked = ((direction == "N" and (row == 0 or (row, col) in horizontal))
                   or (direction == "E" and (col == puzzle["width"] - 1 or (row, col + 1) in vertical))
                   or (direction == "S" and (row == puzzle["height"] - 1 or (row + 1, col) in horizontal))
                   or (direction == "W" and (col == 0 or (row, col) in vertical)))
        if blocked or (row + dr, col + dc) in occupied:
            robots[robot] = [row, col]
            return (row, col) != start
        row, col = row + dr, col + dc

def simulate(puzzle: dict, moves: list[str]) -> tuple[bool, int, dict[str, list[int]]]:
    robots = {name: list(position) for name, position in puzzle["robots"].items()}
    valid = 0
    for token in moves:
        valid += move(puzzle, robots, token)
        if list(puzzle["goal"]) in robots.values():
            return True, valid, robots
    return False, valid, robots

def reward(puzzle: dict, moves: list[str]) -> float:
    solved, valid, robots = simulate(puzzle, moves)
    if solved and len(moves) <= puzzle["max_moves"]:
        return 1.0
    goal_row, goal_col = puzzle["goal"]
    distance = min(abs(row - goal_row) + abs(col - goal_col) for row, col in robots.values())
    return max(0.0, 0.4 - 0.02 * distance + 0.002 * valid)

def show(puzzle: dict) -> None:
    board = [["." for _ in range(puzzle["width"])] for _ in range(puzzle["height"])]
    row, col = puzzle["goal"]
    board[row][col] = "*"
    for name, (row, col) in puzzle["robots"].items():
        board[row][col] = name
    print("\n".join(" ".join(row) for row in board))
    print(f"lesson={puzzle['lesson']} goal={puzzle['goal']} max_moves={puzzle['max_moves']}")

def main() -> int:
    puzzle = load()
    command = sys.argv[1] if len(sys.argv) > 1 else "show"
    if command == "show": show(puzzle); return 0
    if command == "test":
        moves = [item.strip().upper() for item in sys.argv[2].split(",") if item.strip()]
        solved, valid, robots = simulate(puzzle, moves)
        print(json.dumps({"solved": solved, "reward": reward(puzzle, moves), "valid_moves": valid, "robots": robots})); return 0
    if command == "evaluate":
        path = Path(sys.argv[2] if len(sys.argv) > 2 else "/workspace/solution.json")
        try:
            moves = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(moves, list) or not all(isinstance(item, str) for item in moves): raise ValueError("solution must be a JSON array of strings")
            value = reward(puzzle, moves); print(value)
            return 0 if simulate(puzzle, moves)[0] and len(moves) <= puzzle["max_moves"] else 1
        except (OSError, ValueError, json.JSONDecodeError) as error:
            print(f"0.0\n{error}", file=sys.stderr); return 1
    raise ValueError(f"unknown command: {command}")
if __name__ == "__main__": raise SystemExit(main())
