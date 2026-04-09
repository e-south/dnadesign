"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/input_payload_models.py

Resolved input payload models for payload-centric YIU workflows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ResolvedInputPayload:
    input_kind: str
    payload_sequence: str
    payload_label: str | None
    site_label: str | None
    provenance: dict[str, object]
    hit_row: dict[str, Any] | None
    source_artifact_path: Path | None
    sample_workspace_root: Path | None


__all__ = ["ResolvedInputPayload"]
