"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/proteinmpnn_request/constants.py

Stable identifiers for Eco1 ProteinMPNN request materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

DOCS_ROOT = Path("docs/studies/eco1_rt_repack")
STRUCTURE_SOURCES = DOCS_ROOT / "workbench/provenance/structure-sources.yaml"
DEFAULT_OUTPUT_ROOT = Path("outputs/thread/eco1_rt_conservative_v1")
REQUEST_DIR_NAME = "proteinmpnn_request"
PROTEINMPNN_NAME = "chain_a_backbone"
CHAIN_ID = "A"
SCHEMA_ID = "proteinmpnn.fixed_backbone_request"
ARTIFACT_ID = "eco1_rt_conservative_v1.proteinmpnn_request"
CREATED_BY = "dnadesign.studies.units.eco1_rt_repack.operations.materialization.proteinmpnn_request"
