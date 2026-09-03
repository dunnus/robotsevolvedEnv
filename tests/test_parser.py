from robotsevolved_env.puzzle import Puzzle


def test_parse_robots_walls_and_colored_goal() -> None:
    data = {
        "width": 5,
        "height": 4,
        "playerStart": [
            {"top": 1, "left": 1, "colorSignifier": "green"},
            {"top": 2, "left": 1, "colorSignifier": "blue"},
            {"top": 1, "left": 3, "colorSignifier": "yellow"},
            {"top": 2, "left": 3, "colorSignifier": "red"},
        ],
        "wallHorizontal": [{"top": 2, "left": 2, "opacity": 1}],
        "wallVerticle": [
            {"top": 1, "left": 2, "opacity": 1},
            {"top": 2, "left": 2, "opacity": 0},
        ],
        "goal": None,
        "coloredGoals": [{"top": 3, "left": 2, "colorSignifier": "red"}],
    }
    puzzle = Puzzle.from_json(data)
    assert puzzle.robots == ((2, 1), (1, 1), (2, 3), (1, 3))
    assert (2, 2) in puzzle.horizontal_walls
    assert (1, 2) in puzzle.vertical_walls and (2, 2) not in puzzle.vertical_walls
    assert puzzle.goal == (3, 2)
    assert puzzle.target_robot == 2
