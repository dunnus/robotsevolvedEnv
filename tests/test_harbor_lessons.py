from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path


ROOT = Path(__file__).parents[1]
LESSONS_PATH = ROOT / "data" / "lessons" / "lessons.json"


def _oracle_moves(path: Path) -> list[str]:
    match = re.search(r"^\[.*\]$", path.read_text(encoding="utf-8"), re.MULTILINE)
    assert match is not None
    return json.loads(match.group(0))


def test_all_playable_lessons_have_valid_offline_harbor_tasks() -> None:
    snapshot = json.loads(LESSONS_PATH.read_text(encoding="utf-8"))
    lessons = snapshot["lessons"]
    assert len(lessons) == 10
    assert [lesson["source_index"] for lesson in lessons] == [0, 1, 2, 3, 5, 6, 8, 9, 10, 11]

    for lesson in lessons:
        task = ROOT / "harbor" / f"robotsevolved-lesson-{lesson['slug']}"
        config = tomllib.loads((task / "task.toml").read_text(encoding="utf-8"))
        assert config["schema_version"] == "1.4"
        assert config["environment"]["network_mode"] == "no-network"
        assert config["metadata"]["source_id"] == lesson["source_id"]

        puzzle = json.loads((task / "environment" / "puzzle.json").read_text(encoding="utf-8"))
        expected = dict(lesson["puzzle"])
        assert puzzle.pop("lesson") == lesson["name"]
        assert puzzle == expected

        namespace: dict[str, object] = {"__name__": "lesson_verifier"}
        game_path = task / "environment" / "game.py"
        exec(compile(game_path.read_text(encoding="utf-8"), game_path, "exec"), namespace)
        moves = _oracle_moves(task / "solution" / "solve.sh")
        assert moves == lesson["reference_solution"]
        solved, _, _ = namespace["simulate"](puzzle, moves)  # type: ignore[operator]
        assert solved
        assert len(moves) <= puzzle["max_moves"]
        assert namespace["reward"](puzzle, moves) == 1.0  # type: ignore[operator]