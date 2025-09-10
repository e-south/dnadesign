"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/logging_utils.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .utils import ensure_dir, now_iso


def append_jsonl(path: Path, event: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    event = {"ts": now_iso(), **event}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, sort_keys=True) + "\n")
