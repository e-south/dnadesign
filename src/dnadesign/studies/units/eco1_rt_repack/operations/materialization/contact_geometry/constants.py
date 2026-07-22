"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/contact_geometry/constants.py

Constants for Eco1 RT contact-geometry materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.studies.units.eco1_rt_repack.paths import DEFAULT_THREAD_OUTPUT_ROOT

_DOCS_ROOT = Path("docs/studies/eco1_rt_repack")
_STRUCTURE_SOURCES = _DOCS_ROOT / "workbench/provenance/structure-sources.yaml"
_DEFAULT_OUTPUT_ROOT = DEFAULT_THREAD_OUTPUT_ROOT
_CREATED_BY = "dnadesign.studies.units.eco1_rt_repack.operations.materialization.contact_geometry"
_DEFAULT_CREATED_AT = "2026-06-22T00:00:00Z"
_GEOMETRY_BACKEND_ID = "biopython_mmcif_atom_geometry_v1"
_BACKBONE_ATOM_NAMES = {"N", "CA", "C", "O", "OXT"}
_CONTACT_THRESHOLDS = (4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0)
_CHAIN_COUNT_THRESHOLDS = (8.0, 12.0, 15.0, 20.0)


def threshold_id(threshold: float) -> str:
    """Return the stable contact-threshold field suffix."""

    return f"{int(threshold)}a" if float(threshold).is_integer() else str(threshold).replace(".", "_") + "a"
