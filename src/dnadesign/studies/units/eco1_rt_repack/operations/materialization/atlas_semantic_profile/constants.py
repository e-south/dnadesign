"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/atlas_semantic_profile/constants.py

Constants for Eco1 ESM Atlas semantic-profile materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.studies.units.eco1_rt_repack.paths import DEFAULT_THREAD_OUTPUT_ROOT
from dnadesign.thread.adapters.esm_atlas.client import DEFAULT_BASE_URL

DEFAULT_OUTPUT_ROOT = DEFAULT_THREAD_OUTPUT_ROOT
REQUEST_MANIFEST_RELATIVE_PATH = "foldcheck_request/foldcheck_request_manifest.yaml"
FOLDCHECK_REPORT_FILE_NAME = "foldcheck_report.parquet"
STRUCTURE_PREDICTION_ROOT_RELATIVE_PATH = "structure_predictions"
ATLAS_API_BASE_URL = DEFAULT_BASE_URL
ATLAS_API_VERSION = "v1alpha1"
DEFAULT_TOPK_FEATURES = 100
DEFAULT_SEQUENCE_LIMIT = "1"
DEFAULT_ALLOW_FOLD_ON_MISS = False
