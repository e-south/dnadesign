"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/proteinmpnn_sample_ingest/models.py

Typed result models for Eco1 ProteinMPNN sample ingest.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProteinMpnnSampleIngestResult:
    """Paths emitted by Eco1 ProteinMPNN sample ingest."""

    sample_table_path: Path
    backend_run_manifest_path: Path
