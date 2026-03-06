"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/infer/input_parsing.py

Shared parsing helpers for infer CLI text and id inputs.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional


def read_ids_arg(ids: Optional[str]) -> Optional[List[str]]:
    if not ids:
        return None
    path = Path(ids)
    if path.exists():
        text = path.read_text().strip()
        if "\n" in text:
            return [line.strip() for line in text.splitlines() if line.strip()]
        return [item.strip() for item in text.split(",") if item.strip()]
    return [item.strip() for item in ids.split(",") if item.strip()]


def load_nonempty_lines(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]
