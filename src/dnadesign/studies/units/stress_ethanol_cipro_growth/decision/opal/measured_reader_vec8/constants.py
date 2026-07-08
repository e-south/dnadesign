"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/measured_reader_vec8/constants.py

Defines stress OPAL campaign constants for measured Reader vec8 staging.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

STRESS_CAMPAIGN_SLUGS: tuple[str, ...] = (
    "secg_ethanol_rf_sfxi_topn",
    "secg_cipro_rf_sfxi_topn",
    "secg_and_rf_sfxi_topn",
)

BATCH0_HANDOFF_ID = "stress-opal-batch0-sfxi-v1"
READER_VEC8_RECORD_ID = "sfxi_vec8/vec8"
READER_EVIDENCE_SCHEMA_VERSION = "stress_ethanol_cipro_growth.reader_evidence.v1"
READER_EVIDENCE_FILENAME = "reader_evidence_manifest.json"
X_COLUMN = "latentdna__evo2_7b__context_anchor_mean_bidir_concat"

READER_EVIDENCE_PLOT_RECORD_IDS: tuple[str, ...] = (
    "plot:raw_kinetics",
    "plot:intensity_overview",
    "plot:sfxi_vec8_heatmap",
)

READER_EVIDENCE_PLOT_LABELS: dict[str, str] = {
    "plot:raw_kinetics": "raw_kinetics",
    "plot:intensity_overview": "intensity_overview",
    "plot:sfxi_vec8_heatmap": "sfxi_vec8_heatmap",
}

VEC8_COLUMNS: tuple[str, ...] = (
    "v00",
    "v10",
    "v01",
    "v11",
    "y00_star",
    "y10_star",
    "y01_star",
    "y11_star",
)

READER_VEC8_REQUIRED_COLUMNS: tuple[str, ...] = (
    "design_id",
    "time_selected_h",
    "reference_design_id",
    "intensity_log2_offset_delta",
    "r_logic",
    *VEC8_COLUMNS,
    "flat_logic",
)

OPAL_INGEST_COLUMNS: tuple[str, ...] = (
    "id",
    "sequence",
    "design_id",
    "synthesis_name",
    "reader_experiment_id",
    "time_selected_h",
    "intensity_log2_offset_delta",
    *VEC8_COLUMNS,
)

TARGET_TIME_H = 12.0
