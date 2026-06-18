"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/densegen_axis_probe/tfbs/stage_b/collection.py

Stage B campaign-collection ontology for TFBS learnability review.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any


def stage_b_collection_manifest(
    *,
    split_id: str,
    seed: int,
    control_pair_label: str = "Sequence-matched metadata vs row-shuffled control",
    positive_role_label: str = "Sequence-matched metadata",
    control_role_label: str = "Row-shuffled control",
) -> dict[str, Any]:
    """Return the OPAL campaign_collection.v2 manifest for Stage B sentinel review."""

    return {
        "schema_version": "opal.campaign_collection.v2",
        "collection_id": f"densegen_tfbs_stage_b_exact_budget_{split_id}_seed{int(seed)}",
        "dimensions": [
            {"id": "target", "label": "TFBS label"},
            {"id": "label_oracle_kind", "label": "Label source"},
            {"id": "label_family_id", "label": "Label family"},
            {"id": "label_split_id", "label": "Split"},
            {"id": "seed", "label": "Seed"},
        ],
        "relationships": [
            {
                "id": "positive_vs_null",
                "kind": "control_pair",
                "label": control_pair_label,
                "role_dimension": "label_oracle_kind",
                "left_role": "positive",
                "left_role_label": positive_role_label,
                "right_role": "null",
                "right_role_label": control_role_label,
                "match_on": ["target", "label_family_id", "label_split_id", "seed"],
                "replicate_on": ["seed"],
            }
        ],
        # The TFBS probe endpoint is realized label enrichment, not
        # OPAL's selected predicted score. Study-owned realized-label review
        # artifacts register first-class notebook visuals after execution.
        "comparison_views": [],
    }
