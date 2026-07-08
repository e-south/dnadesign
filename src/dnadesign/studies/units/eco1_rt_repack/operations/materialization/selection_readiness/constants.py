"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/selection_readiness/constants.py

Stable identifiers for Eco1 panel-selection materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes.constants import (
    DEFAULT_DESIGN_CLASSES_ROOT,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes.constants import (
    DEFAULT_SOURCE_OUTPUT_ROOT as _DEFAULT_SOURCE_OUTPUT_ROOT,
)

DEFAULT_OUTPUT_ROOT = DEFAULT_DESIGN_CLASSES_ROOT
DEFAULT_SOURCE_OUTPUT_ROOT = _DEFAULT_SOURCE_OUTPUT_ROOT
DEFAULT_SELECTION_DIR_NAME = "selection"
FEASIBILITY_REPORT_FILE_NAME = "feasibility_report.parquet"
CANDIDATE_TRIAGE_TABLE_FILE_NAME = "candidate_triage_table.parquet"
LOCAL_STRUCTURE_REGION_METRICS_FILE_NAME = "local_structure_region_metrics.parquet"
LOCAL_STRUCTURE_THRESHOLD_SENSITIVITY_FILE_NAME = "local_structure_threshold_sensitivity.parquet"
REGION_MSA_SUPPORT_FILE_NAME = "region_msa_support.parquet"
PRIMARY_PANEL_SELECTION_TRACE_FILE_NAME = "primary_panel_selection_trace.parquet"
CANDIDATE_SELECTION_PANEL_FILE_NAME = "candidate_selection_panel.parquet"
CANDIDATE_HANDOFF_SEQUENCE_CSV_FILE_NAME = "candidate_handoff_sequences.csv"
MANIFEST_FILE_NAME = "selection_readiness_manifest.yaml"
PLOTS_DIR_NAME = "plots"
CREATED_BY = "dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness"
DEFAULT_CREATED_AT = "2026-07-02T00:00:00Z"
FEASIBILITY_POLICY_ID = "eco1_rt_full_gene_feasibility_v1"
SELECTION_POLICY_ID = "eco1_rt_primary_conservative_panel_v1"
CODON_POLICY_ID = "protein_sequence_only_no_codon_design_v1"
SAE_WINDOW_SELECTION_THRESHOLD = 0.005
ALLOWED_FOLD_CLASSES = {"strong_fold_preserved"}
PRIMARY_PANEL_SIZE = 6
AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
