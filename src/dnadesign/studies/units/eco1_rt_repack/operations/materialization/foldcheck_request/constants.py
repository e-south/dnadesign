"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/foldcheck_request/constants.py

Constants for Eco1 fold-check request materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.studies.units.eco1_rt_repack.paths import DEFAULT_THREAD_OUTPUT_ROOT

DEFAULT_OUTPUT_ROOT = DEFAULT_THREAD_OUTPUT_ROOT
REQUEST_DIR_NAME = "foldcheck_request"
ARTIFACT_ID = "eco1_rt_conservative_v1.foldcheck_request"
CREATED_BY = "dnadesign.studies.units.eco1_rt_repack.operations.materialization.foldcheck_request"
BACKEND_KIND = "colabfold"
RUNTIME_KIND = "alphafold_family_colabfold"
EXECUTION_STATUS = "planned_not_run"
WT_SEQUENCE_ID = "wild_type"
REFERENCE_STRUCTURE_ID = "ec86kit_7v9u_protomer1"
THRESHOLD_POLICY_ID = "eco1_rt_foldcheck_thresholds_v1"
THRESHOLD_VALUES = {
    "requires_wt_baseline": True,
    "candidate_thresholds_deferred_until_runtime_baseline": True,
}
STORAGE_POLICY = {
    "raw_fold_outputs": "external_runtime_storage",
    "preferred_runtime_locus": "bu_scc_project_storage",
    "usr_sync_scope": "compact_manifests_and_normalized_reports",
}
