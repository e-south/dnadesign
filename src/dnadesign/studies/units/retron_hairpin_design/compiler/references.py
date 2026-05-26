"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/compiler/references.py

Retron MSD design-reference compilation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.contracts.sequence import MsdDesignCatalogV1, MsdDesignReferenceV1

from ..catalog.msd_ids import parse_msd_construct_label
from ..catalog.registry import load_retron_msd_registry
from .exceptions import RetronMsdCompilerError


def build_msd_design_reference(
    label: str,
    *,
    study_dir: str | Path,
    allow_non_ligatable_s0: bool = False,
) -> MsdDesignReferenceV1:
    parsed = parse_msd_construct_label(label, allow_non_ligatable_s0=allow_non_ligatable_s0)
    registry = load_retron_msd_registry(study_dir)
    return registry.build_reference(parsed)


def compile_msd_design_catalog(
    labels: list[str],
    *,
    study_dir: str | Path,
    allow_non_ligatable_s0: bool = False,
) -> MsdDesignCatalogV1:
    if not labels:
        raise RetronMsdCompilerError("Provide at least one construct label.")
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise RetronMsdCompilerError(f"Duplicate construct label(s): {', '.join(duplicates)}")
    references = [
        build_msd_design_reference(
            label,
            study_dir=study_dir,
            allow_non_ligatable_s0=allow_non_ligatable_s0,
        )
        for label in labels
    ]
    return MsdDesignCatalogV1(records=references)


__all__ = ["build_msd_design_reference", "compile_msd_design_catalog"]
