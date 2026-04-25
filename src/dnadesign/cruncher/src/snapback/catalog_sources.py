"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/snapback/catalog_sources.py

Shared source-label helpers for snapback catalog resolution.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence


def catalog_source_label(*, preset_ids: Sequence[str], resolved_paths: Sequence[Path]) -> str:
    labels: list[str] = []
    labels.extend(f"preset:{preset_id}" for preset_id in preset_ids if str(preset_id).strip())
    labels.extend(str(Path(path)) for path in resolved_paths)
    return ", ".join(labels) if labels else "resolved_catalog"


__all__ = ["catalog_source_label"]
