"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/densegen_axis_probe/core/constants.py

Study-owned DenseGen plan-logic OPAL probe package.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

STUDY_ID = "stress_ethanol_cipro_growth"
ORACLE_ID = "densegen_plan_logic4_v1"
NULL_ORACLE_ID = "permuted_densegen_plan_logic4_v1"
DEFAULT_SEED = 7
DEFAULT_SUITE_ID = "densegen_motif_qa_k12_s3_v1"
DEFAULT_SUITE_SEEDS = (7, 17, 29)
DEFAULT_INITIAL_LABELS = 12
DEFAULT_TOP_K = 12
RUN_STAGES = ("materialize", "validate", "init", "ingest", "run", "status")
ACTIVE_LABEL_FAMILY_ID = "densegen_plan_logic4"
ACTIVE_LABEL_FAMILY_IDS = ("densegen_plan_logic4", "tf_family_count")
PASSIVE_LABEL_FAMILY_IDS = ("tf_family_presence", "densegen_plan_class")

STATE_ORDER = ("baseline_or_no_stress", "ethanol", "ciprofloxacin", "ethanol_plus_ciprofloxacin")
DENSEGEN_PLAN_LOGIC4_COLUMNS = ("v00", "v10", "v01", "v11")
DENSEGEN_PLAN_LOGIC4_DISPLAY_LABELS = ("No stress", "Ethanol", "Cipro", "Ethanol + Cipro")

AXIS_CLASS_TO_LOGIC4: dict[str, list[int]] = {
    "background_only": [0, 0, 0, 0],
    "ethanol_only": [0, 1, 0, 1],
    "cipro_only": [0, 0, 1, 1],
    "dual_axis_and": [0, 0, 0, 1],
}

PLAN_TO_AXIS_CLASS: dict[str, str] = {
    "background_only": "background_only",
    "ethanol": "ethanol_only",
    "ciprofloxacin": "cipro_only",
    "ethanol_ciprofloxacin": "dual_axis_and",
}
AXIS_CLASS_TO_DENSEGEN_PLAN_CLASS: dict[str, str] = {
    "background_only": "background_only",
    "ethanol_only": "ethanol",
    "cipro_only": "ciprofloxacin",
    "dual_axis_and": "ethanol_ciprofloxacin",
}

CAMPAIGNS: dict[str, dict[str, Any]] = {
    "cipro": {
        "source_config": "src/dnadesign/opal/campaigns/secg_rmf_greedy/configs/campaign.yaml",
        "target_class": "cipro_only",
        "target_logic4": [0, 0, 1, 1],
    },
    "ethanol": {
        "source_config": "src/dnadesign/opal/campaigns/secg_rmf_greedy/configs/campaign.yaml",
        "target_class": "ethanol_only",
        "target_logic4": [0, 1, 0, 1],
    },
    "dual": {
        "source_config": "src/dnadesign/opal/campaigns/secg_rmf_greedy/configs/campaign.yaml",
        "target_class": "dual_axis_and",
        "target_logic4": [0, 0, 0, 1],
    },
}

SPLITS = ("random_id", "leave_sigma35_variant")
ORACLES = (ORACLE_ID, NULL_ORACLE_ID)

CANDIDATE_RECORDS = Path("src/dnadesign/usr/datasets/usr_prom_eth_cip_opal_candidates/records.parquet")
DENSEGEN_SIDECAR = Path("src/dnadesign/usr/datasets/usr_prom_eth_cip_anchor/_derived/densegen.parquet")
SHARED_OBSERVED_LABEL_SIDECAR = Path(
    "src/dnadesign/usr/datasets/usr_prom_eth_cip_opal_candidates/_opal/observed_labels.parquet"
)
RUN_ROOT_PREFIX = Path(".var") / "studies" / STUDY_ID / "opal_densegen_axis_probe"
X_COLUMN = "latentdna__evo2_7b__context_anchor_mean_bidir_concat"
SCRATCH_DATASET = "opal_densegen_axis_probe_candidates"

FORBIDDEN_PREFIXES = ("latentdna__", "infer__")
FORBIDDEN_EXACT_COLUMNS = {
    "umap_x",
    "umap_y",
    "cluster",
    "cluster_label",
    "opal_prediction",
    "opal_selection",
}

QUALITY_FLAGS = (
    "ok",
    "missing_used_tfbs_detail",
    "malformed_used_tfbs_detail",
    "missing_sigma35_variant",
    "plan_axis_mismatch",
    "unsupported_plan",
)
