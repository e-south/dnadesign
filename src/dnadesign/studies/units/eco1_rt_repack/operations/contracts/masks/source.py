"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/masks/source.py

Manual mask-authority source contract helpers for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.constants import _DOCS_ROOT


def load_manual_mask_authority_source(repo_root: Path) -> dict[str, Any]:
    """Load the checked-in manual mask-authority ontology source."""

    source_path = repo_root / _DOCS_ROOT / "workbench/ontology/manual-mask-authority.yaml"
    loaded = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected YAML mapping at {source_path}")
    return loaded


def candidate_prior_positions_from_source(authority_source: Mapping[str, Any]) -> set[int]:
    """Return candidate-prior positions declared by the ontology source."""

    positions: set[int] = set()
    candidate_sets = authority_source.get("candidate_authority_sets")
    if not isinstance(candidate_sets, list):
        return positions
    for candidate_set in candidate_sets:
        if not isinstance(candidate_set, Mapping):
            continue
        if candidate_set.get("policy") != "candidate_prior_not_mask_authoritative":
            continue
        residues = candidate_set.get("residues")
        if not isinstance(residues, list):
            continue
        for residue in residues:
            if isinstance(residue, Mapping) and isinstance(residue.get("canonical_position"), int):
                positions.add(int(residue["canonical_position"]))
    return positions
