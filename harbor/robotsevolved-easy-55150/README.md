# Robots Evolved Harbor task

This Harbor task asks an agent to solve one locally stored sliding-robot puzzle and write a move sequence to `/workspace/solution.json`. The task has no network access and never contacts or controls the source website.

## Environment

- Harbor task schema 1.4
- Python 3.13 slim container
- One CPU, 512 MB RAM, 1 GB storage
- 300-second agent timeout
- Offline puzzle snapshot and `robot-game` exploration CLI

## Verification

The verifier independently simulates the submitted moves. A valid solution of at most 20 moves earns at least `0.5`; the seven-move reference solution earns `1.0`. Unsolved attempts receive a small progress reward based on final Manhattan distance.

## Layout

- `instruction.md`: agent task and output contract
- `task.toml`: Harbor configuration with `network_mode = "no-network"`
- `environment/`: container, puzzle snapshot, and simulator
- `solution/solve.sh`: Oracle answer
- `tests/test.sh`: writes `/logs/verifier/reward.txt`

## Run

From a Harbor checkout or installation, run the task path with the Oracle first, then with the desired agent/model. For example: `harbor run -p ./harbor/robotsevolved-easy-55150 -a oracle`.
