"""Create a self-contained HTML visualization of a rollout."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Sequence

from robotsevolved_env.data import load_record
from robotsevolved_env.puzzle import Puzzle, Position
from robotsevolved_env.rollout import ReplayResult, replay_moves

COLORS = ("#4169e1", "#228b22", "#b22222", "#ff8c00")


def _svg(puzzle: Puzzle, robots: Sequence[Position], *, cell: int = 34) -> str:
    width, height = puzzle.width * cell, puzzle.height * cell
    elements = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Puzzle board">',
        f'<rect width="{width}" height="{height}" fill="#eef0ea"/>',
    ]
    for row in range(puzzle.height + 1):
        elements.append(f'<line x1="0" y1="{row * cell}" x2="{width}" y2="{row * cell}" class="grid"/>')
    for col in range(puzzle.width + 1):
        elements.append(f'<line x1="{col * cell}" y1="0" x2="{col * cell}" y2="{height}" class="grid"/>')
    goal_row, goal_col = puzzle.goal
    elements.append(
        f'<circle cx="{goal_col * cell + cell / 2}" cy="{goal_row * cell + cell / 2}" r="8" fill="none" stroke="#111" stroke-width="4"/>'
    )
    for row, col in puzzle.horizontal_walls:
        elements.append(f'<line x1="{col * cell}" y1="{row * cell}" x2="{(col + 1) * cell}" y2="{row * cell}" class="wall"/>')
    for row, col in puzzle.vertical_walls:
        elements.append(f'<line x1="{col * cell}" y1="{row * cell}" x2="{col * cell}" y2="{(row + 1) * cell}" class="wall"/>')
    for index, (row, col) in enumerate(robots):
        x, y = col * cell + cell / 2, row * cell + cell / 2
        elements.append(f'<circle cx="{x}" cy="{y}" r="12" fill="{COLORS[index]}"/>')
        elements.append(f'<text x="{x}" y="{y + 5}" text-anchor="middle" fill="white">{"BGRY"[index]}</text>')
    elements.append("</svg>")
    return "".join(elements)


def save_rollout_html(puzzle: Puzzle, rollout: ReplayResult, output: str | Path) -> Path:
    """Write an interactive, dependency-free rollout viewer."""
    frames = [_svg(puzzle, robots) for robots in rollout.states]
    labels = ["Initial"] + [
        f"Step {index}: {html.escape(move)} | reward {rollout.rewards[index - 1]:.2f}"
        for index, move in enumerate(rollout.moves, 1)
    ]
    frame_markup = "".join(
        f'<section class="frame" data-index="{index}">{svg}</section>'
        for index, svg in enumerate(frames)
    )
    labels_json = json.dumps(labels).replace("</", "<\\/")
    document = f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><title>Robots Evolved rollout</title>
<style>
body{{font-family:system-ui;background:#141820;color:#eef2f6;margin:0;padding:24px}}
main{{max-width:760px;margin:auto;background:#202631;padding:24px;border-radius:16px}}
.viewer{{background:white;padding:12px;border-radius:12px}} .frame{{display:none}} .frame.active{{display:block}}
svg{{display:block;width:100%;max-height:70vh}} .grid{{stroke:#c8ccc4;stroke-width:1}} .wall{{stroke:#101317;stroke-width:5;stroke-linecap:round}}
.controls{{display:flex;gap:10px;align-items:center;margin-top:16px}} input{{flex:1}} button{{padding:8px 14px}}
.status{{color:#9ee6b3;font-weight:700}}
</style><body><main><h1>Rollout viewer</h1>
<p class="status">Solved: {str(rollout.solved).lower()} · moves: {len(rollout.moves)} · invalid: {rollout.invalid_moves}</p>
<div class="viewer">{frame_markup}</div><h2 id="label"></h2>
<div class="controls"><button id="prev">Previous</button><input id="step" type="range" min="0" max="{len(frames) - 1}" value="0"><button id="next">Next</button><button id="play">Play</button></div>
<script>const labels={labels_json},frames=[...document.querySelectorAll('.frame')],slider=document.querySelector('#step'),label=document.querySelector('#label'),prev=document.querySelector('#prev'),next=document.querySelector('#next'),play=document.querySelector('#play');let timer;
function show(i){{i=Math.max(0,Math.min(frames.length-1,i));slider.value=i;frames.forEach((f,j)=>f.classList.toggle('active',j===i));label.textContent=labels[i]}}
prev.onclick=()=>show(+slider.value-1);next.onclick=()=>show(+slider.value+1);slider.oninput=()=>show(+slider.value);play.onclick=()=>{{clearInterval(timer);show(0);timer=setInterval(()=>{{if(+slider.value>=frames.length-1)return clearInterval(timer);show(+slider.value+1)}},650)}};show(0);</script></main></body></html>"""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--puzzle", type=Path, required=True, help="Local curated puzzle record")
    parser.add_argument("--moves", help="Comma-separated moves; defaults to the record's reference solution")
    parser.add_argument("--output", type=Path, default=Path("artifacts/rollout.html"))
    args = parser.parse_args()
    record = load_record(args.puzzle)
    puzzle = Puzzle.from_json(record["puzzle"])
    moves = args.moves.split(",") if args.moves else record.get("reference_solution", [])
    result = replay_moves(puzzle, moves)
    print(save_rollout_html(puzzle, result, args.output))


if __name__ == "__main__":
    main()