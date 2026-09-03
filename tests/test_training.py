from __future__ import annotations

import json

from robotsevolved_env.observation import text_observation, training_prompt
from robotsevolved_env.puzzle import Puzzle
from robotsevolved_env.trl_train import format_reward, puzzle_reward
from robotsevolved_env.visualization import save_rollout_html
from robotsevolved_env.rollout import replay_moves


def puzzle() -> Puzzle:
    return Puzzle(
        width=5,
        height=5,
        robots=((2, 1), (2, 3), (0, 0), (4, 4)),
        horizontal_walls=frozenset((row, col) for col in range(5) for row in (0, 5)),
        vertical_walls=frozenset((row, col) for row in range(5) for col in (0, 5)),
        goal=(2, 2),
    )


def encoded_puzzle() -> str:
    value = puzzle()
    return json.dumps({
        "width": value.width,
        "height": value.height,
        "playerStart": [
            {"top": row, "left": col, "colorSignifier": color}
            for (row, col), color in zip(value.robots, ("blue", "green", "red", "yellow"))
        ],
        "wallHorizontal": [
            {"top": row, "left": col, "opacity": 1}
            for row, col in value.horizontal_walls
        ],
        "wallVerticle": [
            {"top": row, "left": col, "opacity": 1}
            for row, col in value.vertical_walls
        ],
        "goal": {"top": value.goal[0], "left": value.goal[1]},
    })


def test_text_observation_discloses_complete_state() -> None:
    observed = text_observation(puzzle())
    assert "B=(2,1)" in observed
    assert "Goal: (2,2), target=any robot" in observed
    assert "Horizontal walls:" in observed and "Vertical walls:" in observed
    assert "Return only a JSON array" in training_prompt(puzzle())


def test_trl_rewards_are_local_and_programmatic() -> None:
    assert puzzle_reward(['["BE"]'], [encoded_puzzle()], [20]) == [0.9875]
    assert puzzle_reward(["not json"], [encoded_puzzle()], [20]) == [0.0]
    assert format_reward(['["BE"]', "bad"]) == [0.1, 0.0]


def test_html_rollout_contains_all_frames(tmp_path) -> None:
    result = replay_moves(puzzle(), ["BE"])
    output = save_rollout_html(puzzle(), result, tmp_path / "rollout.html")
    document = output.read_text(encoding="utf-8")
    assert "Rollout viewer" in document
    assert document.count('class="frame"') == 2
    assert "Solved: true" in document