# RobotsEvolved RL Environment

A local [Gymnasium](https://gymnasium.farama.org/) model of sliding-robot puzzle mechanics, plus an offline [Harbor](https://github.com/harbor-framework/harbor) agent-evaluation task. RobotsEvolved.com was used only to understand the rules and collect static puzzle snapshots. No runtime component contacts or controls the website.

## Mechanics

Four robots slide in cardinal directions until stopped by a wall or another robot. A puzzle is solved when any robot reaches a wildcard goal, or when the matching robot reaches a colored goal.

The action space is `Discrete(16)`:

```text
action = robot * 4 + direction
robots:     0 blue, 1 green, 2 red, 3 yellow
directions: 0 up,   1 right, 2 down, 3 left
```

### What the agent observes

There are two deliberately different policy interfaces:

1. **Gymnasium tensor policy** — `reset()` and `step()` return a dictionary containing:
    - `robots`: `(4, 2)` row/column coordinates in blue, green, red, yellow order.
    - `goal`: `(2,)` row/column coordinate.
    - `target_robot`: `-1` for a wildcard goal or robot index `0..3`.
    - `horizontal_walls`: `(height + 1, width)` binary crossing barriers.
    - `vertical_walls`: `(height, width + 1)` binary crossing barriers.
    - `action_mask`: 16 binary values; zero means the action cannot move that robot.
2. **TRL language-model policy** — the prompt contains the same complete state as deterministic text: dimensions, all robot coordinates, goal/target, and both wall lists with coordinate semantics. The model emits one JSON move sequence such as `["GW","GS"]`, which is parsed and replayed locally for reward.

The Harbor task is interactive instead: the agent reads its instruction and may call the local `robot-game show` and `robot-game test` commands. It still has no website or network access.

## Setup

Requires Python 3.11 or newer.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
pytest
```

## Use offline boards

```python
from robotsevolved_env import RobotsEvolvedEnv

env = RobotsEvolvedEnv(width=12, height=12, max_steps=256, reward_mode="dense")
observation, info = env.reset(seed=42)

while True:
    valid_actions = info["action_mask"].nonzero()[0]
    action = int(env.np_random.choice(valid_actions))
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

Generated boards are deterministic under `reset(seed=...)`, but are not guaranteed to be solvable. For a curriculum, reject boards an agent or solver cannot solve.

## Use collected local data

```python
import json
from pathlib import Path

from robotsevolved_env import Puzzle, RobotsEvolvedEnv

record = json.loads(Path("data/puzzles/easy-55150.json").read_text())
puzzle = Puzzle.from_json(record["puzzle"])
env = RobotsEvolvedEnv(puzzle, render_mode="ansi")
observation, info = env.reset()
print(env.render())
```

The normalized snapshot records provenance separately from puzzle state. Dataset acquisition is intentionally outside the environment package. If more examples are collected, do so as a separate, rate-limited curation step permitted by the site's terms; review and store them locally before running agents. **Do not let agents access the site or automate score/leaderboard submissions.**

## Harbor task

The task in `harbor/robotsevolved-easy-55150` follows Harbor schema 1.4 with `instruction.md`, `task.toml`, an offline Docker environment, Oracle solution, and reward-producing verifier. The container sets `network_mode = "no-network"`; agents inspect and test moves through the local `robot-game` CLI.

The `harbor` directory also contains one self-contained task for each of the 10 playable boards in the website's Lessons mode: four beginner, two intermediate, and four advanced tasks. The two duplicate Moving records used as website section separators are intentionally excluded. See `harbor/README.md` for the complete task matrix. The curated source snapshot, descriptions, and verified reference solutions are stored in `data/lessons/lessons.json`.

From a Harbor installation, run the Oracle validation first:

```powershell
harbor run -p ./harbor/robotsevolved-easy-55150 -a oracle
```

## Hill-climbing baseline

After installation:

```powershell
robotsevolved-hillclimb --iterations 10000 --length 64 --seed 7
```

The baseline mutates a fixed-length sequence of the 16 discrete actions and retains non-worse candidates. It is intentionally simple; PPO/DQN with action masking or a search-based solver will generally be stronger.

## Visualize rollouts

Generate a self-contained interactive HTML replay from a local puzzle and its stored reference solution:

```powershell
robotsevolved-rollout --puzzle data/puzzles/easy-55150.json --output artifacts/easy-55150.html
```

To inspect another rollout, pass moves explicitly:

```powershell
robotsevolved-rollout --puzzle data/puzzles/easy-55150.json --moves GW,GS,RN,RW,BN,BW,BS
```

The viewer draws the board, walls, goal, and color-coded robots for every step, with previous/next, slider, and autoplay controls. It embeds no external scripts or assets.

## Train with TRL

TRL trains a language model, not the 16-action tensor policy. This project uses `GRPOTrainer`: the policy generates JSON move sequences and two local reward functions score puzzle progress/correctness and output format. All puzzle data and reward computation remain offline.

Install the optional training stack in a GPU-capable environment:

```powershell
python -m pip install -e ".[train]"
```

Download or provision the model separately, then pass a **local model directory**. Model loading uses `local_files_only=True`:

```powershell
accelerate launch -m robotsevolved_env.trl_train `
    --model C:/models/Qwen2.5-0.5B-Instruct `
    --data-dir data/puzzles `
    --output-dir artifacts/trl-grpo `
    --steps 500
```

For a small CPU wiring check, add `--cpu --steps 1`; actual GRPO training is normally GPU-intensive. `--batch-size` must be divisible by `--num-generations` for a single process. With only one curated puzzle this is an overfitting smoke test, not a useful general policy—collect a permitted, diverse local dataset before substantive training.

## Rewards

- Solved: `+10`
- Invalid/no-movement action: `-1`
- Sparse mode, valid move: `-0.01`
- Dense mode, valid move: `-0.05 + 0.1 * (old Manhattan distance - new distance)`

Distance shaping is convenient for hill climbing, but this game often requires temporarily moving away from the goal. Sparse rewards or potential-based shaping may be preferable for unbiased policy learning.

TRL uses a separate bounded sequence reward in `sequence_reward()`: solved sequences receive the largest reward with an efficiency adjustment; unsolved sequences receive limited final-distance shaping; malformed moves and no-ops are penalized. A small independent format reward encourages valid JSON.

## Puzzle format

`Puzzle.from_json()` understands the website's `width`, `height`, `playerStart`/`playerState`, `wallHorizontal`, misspelled `wallVerticle`, `goal`, and `coloredGoals` fields. Walls with `opacity != 1` and switch-controlled walls are treated as open; dynamic Daily Evolution switches are outside this initial environment's scope.
