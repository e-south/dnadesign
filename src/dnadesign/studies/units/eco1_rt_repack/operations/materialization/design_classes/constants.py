"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/design_classes/constants.py

Stable identifiers for Eco1 RT design-class expansion materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.studies.units.eco1_rt_repack.paths import DEFAULT_THREAD_OUTPUT_ROOT

DEFAULT_SOURCE_OUTPUT_ROOT = DEFAULT_THREAD_OUTPUT_ROOT
DEFAULT_DESIGN_CLASSES_ROOT = DEFAULT_THREAD_OUTPUT_ROOT / "design_classes"
DESIGN_CLASS_MANIFEST_FILE_NAME = "design_class_manifest.yaml"
CANDIDATE_POOL_FILE_NAME = "candidate_pool.parquet"
CANDIDATE_POOL_MANIFEST_FILE_NAME = "candidate_pool_manifest.yaml"
FOLDCHECK_REQUEST_DIR_NAME = "foldcheck_request"
BASELINE_CLASS_ID = "eco1_rt_clade9_plurality25_contact5a_v1"
CREATED_BY = "dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes"
DEFAULT_CREATED_AT = "2026-07-01T00:00:00Z"

DOCS_ROOT = Path("docs/studies/eco1_rt_repack")
PROFILE_PATH = DOCS_ROOT / "operations/contract/fixtures/thread/eco1_rt_v1.profile.yaml"
CONSERVATION_SOURCES_PATH = DOCS_ROOT / "workbench/provenance/conservation-sources.yaml"
