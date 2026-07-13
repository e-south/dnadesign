"""Result models for the Eco1 RT Twist handoff."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MaterializedTwistHandoff:
    """Paths emitted by one validated Twist handoff materialization."""

    manifest_path: Path
    twist_csv_path: Path
    fasta_path: Path
    genbank_paths: tuple[Path, ...]
