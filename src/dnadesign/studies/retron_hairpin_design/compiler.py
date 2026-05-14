"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/retron_hairpin_design/compiler.py

Study-local Retron MSD design-reference compiler.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.contracts.sequence import MsdDesignCatalogV1, MsdDesignReferenceV1

from .msd_ids import parse_msd_construct_label
from .registry import load_retron_msd_registry

REFERENCE_FILENAME = "msd_design_reference_v1.json"
CATALOG_FILENAME = "msd_design_catalog_v1.json"


class RetronMsdCompilerError(ValueError):
    """Raised when MSD design-reference compilation cannot proceed safely."""


def build_msd_design_reference(label: str, *, study_dir: str | Path) -> MsdDesignReferenceV1:
    parsed = parse_msd_construct_label(label)
    registry = load_retron_msd_registry(study_dir)
    return registry.build_reference(parsed)


def compile_msd_design_catalog(labels: list[str], *, study_dir: str | Path) -> MsdDesignCatalogV1:
    if not labels:
        raise RetronMsdCompilerError("Provide at least one construct label.")
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise RetronMsdCompilerError(f"Duplicate construct label(s): {', '.join(duplicates)}")
    references = [build_msd_design_reference(label, study_dir=study_dir) for label in labels]
    return MsdDesignCatalogV1(records=references)


def write_msd_design_catalog(catalog: MsdDesignCatalogV1, *, out_dir: str | Path) -> Path:
    root = Path(out_dir).expanduser().resolve()
    assets_dir = root / "assets"
    for record in catalog.records:
        reference_dir = assets_dir / record.msd_design_id
        reference_dir.mkdir(parents=True, exist_ok=True)
        reference_path = reference_dir / REFERENCE_FILENAME
        reference_path.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
    root.mkdir(parents=True, exist_ok=True)
    catalog_path = root / CATALOG_FILENAME
    catalog_path.write_text(catalog.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return catalog_path


__all__ = [
    "CATALOG_FILENAME",
    "REFERENCE_FILENAME",
    "RetronMsdCompilerError",
    "build_msd_design_reference",
    "compile_msd_design_catalog",
    "write_msd_design_catalog",
]
