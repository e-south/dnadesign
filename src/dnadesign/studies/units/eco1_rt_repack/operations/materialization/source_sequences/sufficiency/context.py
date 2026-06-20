"""Input context for Eco1 source-sequence bundle sufficiency checks."""

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
