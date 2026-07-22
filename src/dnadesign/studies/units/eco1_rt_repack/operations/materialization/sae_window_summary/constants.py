"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/sae_window_summary/constants.py

Stable identifiers for Eco1 SAE window-summary materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.studies.units.eco1_rt_repack.paths import DEFAULT_THREAD_OUTPUT_ROOT

DEFAULT_SOURCE_OUTPUT_ROOT = DEFAULT_THREAD_OUTPUT_ROOT
DEFAULT_OUTPUT_ROOT = DEFAULT_THREAD_OUTPUT_ROOT / "design_classes"
DEFAULT_REPORT_ROOT = Path("biohub_esmc")
SUMMARY_FILE_NAME = "sae_feature_window_summary.parquet"
MANIFEST_FILE_NAME = "sae_feature_window_summary_manifest.yaml"
DEFAULT_CREATED_AT = "2026-07-02T00:00:00Z"
METHOD_ID = "biohub_esmc_sae_window_delta_v1"
CREATED_BY = "dnadesign.studies.units.eco1_rt_repack.operations.materialization.sae_window_summary"
INTERPRETATION_LIMIT = (
    "SAE windows compare local model-feature activation summaries to WT. They are review annotations, "
    "not activity measurements, fold validation, or evidence of strand displacement."
)
