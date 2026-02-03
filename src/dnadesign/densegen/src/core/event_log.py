"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/core/event_log.py

Event log parsing helpers for DenseGen runs.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_events(events_path: Path, *, allow_missing: bool = False) -> pd.DataFrame:
    if not events_path.exists():
        if allow_missing:
            return pd.DataFrame(
                columns=["event", "created_at", "input_name", "plan_name", "library_index", "library_hash"]
            )
        raise FileNotFoundError(f"events log not found: {events_path}")
    rows = []
    for line in events_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except Exception as exc:
            raise ValueError(f"Invalid JSON in events log: {events_path}") from exc
        rows.append(payload)
    return pd.DataFrame(rows)
