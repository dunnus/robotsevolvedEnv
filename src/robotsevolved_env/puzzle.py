"""Puzzle model and parsing for locally curated puzzle snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

Position = tuple[int, int]  # row, column
ROBOT_COLORS = ("blue", "green", "red", "yellow")


@dataclass(frozen=True, slots=True)
class Puzzle:
    """Immutable puzzle definition using grid-line wall coordinates."""

    width: int
    height: int
    robots: tuple[Position, Position, Position, Position]
    horizontal_walls: frozenset[Position]
    vertical_walls: frozenset[Position]
    goal: Position
    target_robot: int | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        if self.width < 2 or self.height < 2:
            raise ValueError("Puzzle dimensions must be at least 2x2")
        if len(set(self.robots)) != 4:
            raise ValueError("The four robot positions must be distinct")
        for row, col in (*self.robots, self.goal):
            if not (0 <= row < self.height and 0 <= col < self.width):
                raise ValueError(f"Position {(row, col)} is outside the board")
        if self.target_robot is not None and not 0 <= self.target_robot < 4:
            raise ValueError("target_robot must be 0..3 or None")

    @classmethod
    def from_json(cls, data: Mapping[str, Any], *, name: str | None = None) -> "Puzzle":
        """Parse the puzzledata format retained in local snapshots."""
        width, height = int(data["width"]), int(data["height"])
        raw_robots = data.get("playerStart", data.get("playerState"))
        if not isinstance(raw_robots, Sequence) or len(raw_robots) != 4:
            raise ValueError("puzzledata must contain four robots in playerStart/playerState")

        by_color: dict[str, Position] = {}
        fallback: list[Position] = []
        for raw in raw_robots:
            pos = (int(raw["top"]), int(raw["left"]))
            fallback.append(pos)
            color = str(raw.get("colorSignifier", "")).lower()
            if color:
                by_color[color] = pos
        robots = tuple(by_color.get(color, fallback[index]) for index, color in enumerate(ROBOT_COLORS))

        horizontal = _parse_walls(data.get("wallHorizontal", ()))
        vertical = _parse_walls(data.get("wallVerticle", data.get("wallVertical", ())))
        horizontal |= frozenset((row, col) for col in range(width) for row in (0, height))
        vertical |= frozenset((row, col) for row in range(height) for col in (0, width))

        goal_raw = data.get("goal")
        colored = data.get("coloredGoals") or ()
        target_robot: int | None = None
        if goal_raw is None:
            if not colored:
                raise ValueError("puzzledata has no goal")
            goal_raw = colored[0]
            color = str(goal_raw.get("colorSignifier", "")).lower()
            if color not in ROBOT_COLORS:
                raise ValueError(f"Unknown colored goal: {color!r}")
            target_robot = ROBOT_COLORS.index(color)

        return cls(
            width=width,
            height=height,
            robots=robots,  # type: ignore[arg-type]
            horizontal_walls=horizontal,
            vertical_walls=vertical,
            goal=(int(goal_raw["top"]), int(goal_raw["left"])),
            target_robot=target_robot,
            name=name,
        )


def _parse_walls(raw_walls: Sequence[Mapping[str, Any]]) -> frozenset[Position]:
    return frozenset(
        (int(wall["top"]), int(wall["left"]))
        for wall in raw_walls
        if float(wall.get("opacity", 1)) == 1 and wall.get("wallType") is None
    )
