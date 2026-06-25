"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/sufficiency/context.py

Input context for Eco1 source-sequence bundle sufficiency checks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SourceSequenceSufficiencyContext:
    """Resolved inputs for source-sequence bundle sufficiency validation."""

    repo_root: Path
    output_root: Path
    source_cache_root: Path
    bundle_root: Path
    conservation_sources_path: Path
    conservation_sources: Mapping[str, Any]
    selected_profile_ids: tuple[str, ...] | None = None
