"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/selection_readiness/constants.py

Stable identifiers for Eco1 panel-selection materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.generation_policies.constants import (
    DEFAULT_GENERATION_POLICIES_ROOT,
)
from dnadesign.studies.units.eco1_rt_repack.paths import DEFAULT_THREAD_OUTPUT_ROOT

DEFAULT_OUTPUT_ROOT = DEFAULT_GENERATION_POLICIES_ROOT
DEFAULT_SOURCE_OUTPUT_ROOT = DEFAULT_THREAD_OUTPUT_ROOT
DEFAULT_SELECTION_DIR_NAME = "selection"
CANDIDATE_TRIAGE_TABLE_FILE_NAME = "candidate_triage_table.parquet"
LOCAL_STRUCTURE_REGION_METRICS_FILE_NAME = "local_structure_region_metrics.parquet"
LOCAL_STRUCTURE_THRESHOLD_SENSITIVITY_FILE_NAME = "local_structure_threshold_sensitivity.parquet"
REGION_MSA_SUPPORT_FILE_NAME = "region_msa_support.parquet"
HYPOTHESIS_PANEL_SELECTION_TRACE_FILE_NAME = "hypothesis_panel_selection_trace.parquet"
CANDIDATE_SELECTION_PANEL_FILE_NAME = "candidate_selection_panel.parquet"
CANDIDATE_HANDOFF_SEQUENCE_CSV_FILE_NAME = "candidate_handoff_sequences.csv"
MANIFEST_FILE_NAME = "selection_readiness_manifest.yaml"
PLOTS_DIR_NAME = "plots"
CREATED_BY = "dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness"
DEFAULT_CREATED_AT = "2026-07-02T00:00:00Z"
CODON_POLICY_ID = "protein_sequence_only_no_codon_design_v1"
SAE_WINDOW_SELECTION_THRESHOLD = 0.005
WANG_ALPHA1_R13_POSITION = 13
WANG_ALPHA1_CONTEXT_POSITIONS = frozenset(range(4, 17))
WANG_ALPHA1_CROSS_PROTOMER_CONTACT_POSITIONS = frozenset({10, 13})
WANG_TESTED_INTERFACE_DISRUPTING_SUBSTITUTION = "R13A"
AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
