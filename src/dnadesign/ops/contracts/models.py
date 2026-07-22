"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/contracts/models.py

Public Ops contract records.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ResumeReadinessPolicy:
    tool: str
    required_record_columns: tuple[str, ...]
    orphan_artifact_markers: tuple[str, ...]


@dataclass(frozen=True)
class USRProducerContract:
    tool: str
    config_path: Path
    run_root: Path | None
    usr_root: Path
    usr_dataset: str
    supports_overlay_parts: bool
    supports_records_parts: bool
    usr_chunk_size: int | None
    records_path: Path | None
    parquet_chunk_size: int | None
    round_robin: bool | None
    max_accepted_per_library: int | None
    generation_total_quota: int | None
