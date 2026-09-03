Solve the sliding-robot puzzle stored locally at `/data/puzzle.json`.

Robots move north (`N`), east (`E`), south (`S`), or west (`W`) and continue until a wall or another robot stops them. Robot letters are `B` (blue), `G` (green), `R` (red), and `Y` (yellow). The wildcard goal is solved when any robot stops on it.

Use the offline helper—there is no network access:

- `robot-game show` prints the board and current puzzle metadata.
- `robot-game test GW,GS,...` simulates a candidate and reports progress.

Write the final answer to `/workspace/solution.json` as a JSON array of move strings, for example `["GW", "GS"]`. Find a valid solution using no more than 20 moves. Do not modify `/data/puzzle.json` or `/opt/robotsevolved/game.py`.
