"""Load curated puzzle records from local JSON files only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from robotsevolved_env.puzzle import Puzzle


def load_record(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate one local dataset record."""
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(record, dict) or not isinstance(record.get("puzzle"), dict):
        raise ValueError(f"{path} is not a puzzle record")
    return record


def load_puzzle(path: str | Path) -> Puzzle:
    """Create a puzzle from one local dataset record."""
    record = load_record(path)
    provenance = record.get("provenance", {})
    name = str(provenance.get("source_id", Path(path).stem))
    return Puzzle.from_json(record["puzzle"], name=name)


def load_records(data_dir: str | Path) -> list[dict[str, Any]]:
    """Load every JSON puzzle record below a local directory."""
    paths = sorted(Path(data_dir).glob("*.json"))
    if not paths:
        raise ValueError(f"No JSON puzzle records found in {data_dir}")
    return [load_record(path) for path in paths]