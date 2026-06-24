"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/surface_accessibility/constants.py

Constants for Eco1 RT surface-accessibility materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

_DOCS_ROOT = Path("docs/studies/eco1_rt_repack")
_STRUCTURE_SOURCES = _DOCS_ROOT / "workbench/provenance/structure-sources.yaml"
_DEFAULT_OUTPUT_ROOT = Path("outputs/thread/eco1_rt_conservative_v1")
_CREATED_BY = "dnadesign.studies.units.eco1_rt_repack.operations.materialization.surface_accessibility"
_DEFAULT_CREATED_AT = "2026-06-22T00:00:00Z"
_SURFACE_BACKEND_ID = "biopython_shrake_rupley_sasa_v1"
_SHRAKE_RUPLEY_N_POINTS = 100
_BACKBONE_ATOM_NAMES = {"N", "CA", "C", "O", "OXT"}
