"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/evaluation/multistate_behavior_cardinality.py

Family-cardinality pressure evidence for the behavior scalar.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from dnadesign.opal import score_multistate_response_behavior

from .multistate_behavior_protocol import MultistateBehaviorShadowProtocol

_STATE_COUNTS = (2, 4, 8, 16)


def build_family_cardinality_pressure(protocol: MultistateBehaviorShadowProtocol) -> pd.DataFrame:
    """Quantify family dilution while preserving the fixed-K campaign boundary."""

    records: list[dict[str, object]] = []
    for state_count in _STATE_COUNTS:
        on_count = state_count // 2
        off_count = state_count - on_count
        state_ids = tuple(f"s{index:02d}" for index in range(state_count))
        mask = tuple([1.0] * on_count + [0.0] * off_count)
        response = np.asarray([2.0] * on_count + [0.0] * off_count)
        signal = np.asarray([2.0] * on_count + [3.0] + [-2.0] * (off_count - 1))
        values = np.concatenate([response, signal]).reshape(1, -1)
        score = score_multistate_response_behavior(
            values,
            state_ids=state_ids,
            target_mask=mask,
            normalization={"response_scale": 1.0, "signal_scale": 1.0},
        )
        weak_label = f"off_signal_suppression:{state_ids[on_count]}"
        weak_index = score.coordinate_labels.index(weak_label)
        response_pair_count = on_count * off_count
        records.append(
            {
                "state_count": state_count,
                "on_state_count": on_count,
                "off_state_count": off_count,
                "response_pair_count": response_pair_count,
                "analytic_global_maximum_soft_vs_hard_gap": math.log(3.0 * response_pair_count),
                "weak_coordinate_analytic_soft_vs_hard_gap": math.log(3.0 * off_count),
                "realizable_behavior_score": float(score.behavior_score[0]),
                "realizable_hard_bottleneck": float(score.hard_bottleneck_clearance[0]),
                "realizable_soft_vs_hard_gap": float(score.behavior_score[0] - score.hard_bottleneck_clearance[0]),
                "weak_coordinate_bottleneck_weight": float(score.coordinate_weights[0, weak_index]),
                "weak_coordinate_label": weak_label,
                "strong_clearance": 2.0,
                "weak_clearance": -3.0,
                "comparison_scope": "fixed_state_contract_only",
                "hard_bottleneck_role": protocol.hard_bottleneck_role,
                "protocol_id": protocol.protocol_id,
                "protocol_source_sha256": f"sha256:{protocol.source_sha256}",
                "evidence_role": "family_cardinality_pressure_not_campaign_parameter_tuning",
            }
        )
    return pd.DataFrame.from_records(records)


__all__ = ["build_family_cardinality_pressure"]
