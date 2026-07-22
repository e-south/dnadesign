"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/response_metastudy/test_observed_sfxi_replay.py

Contracts for the historical observed-label SFXI replay.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dnadesign.opal import SFXIScoringConfig, score_vec8
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.evaluation import (
    ObservedSfxiViewContext,
    build_observed_sfxi_decomposition,
    summarize_observed_sfxi_decomposition,
)

VEC8_COLUMNS = ("v00", "v10", "v01", "v11", "y00_star", "y10_star", "y01_star", "y11_star")


def test_observed_sfxi_replay_scores_only_explicit_historical_sfxi_vectors() -> None:
    source, labels, contexts, active = _fixture()

    detail = build_observed_sfxi_decomposition(
        source,
        labels,
        view_contexts=contexts,
        active_identities=active,
        top_k=2,
    )
    summary = summarize_observed_sfxi_decomposition(detail)

    assert len(detail) == len(source) * len(contexts)
    assert detail.groupby("selection_view_id")["id"].nunique().to_dict() == {
        "and": len(source),
        "ciprofloxacin": len(source),
        "ethanol": len(source),
    }
    assert detail.groupby("selection_view_id")["is_highest_observed_sfxi"].sum().eq(2).all()
    assert detail.groupby("selection_view_id")["in_promoted_response_window_corpus"].sum().eq(len(active)).all()
    assert detail.groupby("selection_view_id")["is_sensor_control"].sum().eq(2).all()
    np.testing.assert_allclose(
        detail["sfxi"].to_numpy(dtype=float),
        detail["logic_fidelity"].to_numpy(dtype=float) * detail["effect_scaled"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1.0e-12,
    )
    assert np.allclose(detail["denom_persisted"], detail["denom_recomputed"], rtol=0.0, atol=1.0e-12)
    assert set(detail["source_y_contract"]) == {"sfxi_vec8"}
    assert set(detail["promoted_response_window_corpus_status"]) == {"verified"}
    assert set(detail["target_mask"]) == {"0|1|0|1", "0|0|1|1", "0|0|0|1"}
    assert set(summary["selection_view_id"]) == {"ethanol", "ciprofloxacin", "and"}
    assert len(summary) == len(contexts) * (1 + source["reader_experiment_id"].nunique() + 1)
    assert set(summary["sensitivity_scope"]) == {
        "all_observed_labels",
        "leave_one_experiment_out",
        "es_designs_only",
    }
    assert set(summary.loc[summary["sensitivity_scope"] == "all_observed_labels", "candidate_count"]) == {len(source)}
    assert set(summary.loc[summary["sensitivity_scope"] == "es_designs_only", "candidate_count"]) == {len(source) - 2}
    assert summary["correlation_defined"].all()
    assert summary.filter(like="spearman").apply(np.isfinite).all().all()


def test_observed_sfxi_replay_preserves_independent_label_truth_gate() -> None:
    source, labels, contexts, _active = _fixture()

    detail = build_observed_sfxi_decomposition(
        source,
        labels,
        view_contexts=contexts,
        active_identities=None,
        top_k=2,
    )

    assert detail["in_promoted_response_window_corpus"].isna().all()
    assert set(detail["promoted_response_window_corpus_status"]) == {"not_available"}


def test_observed_sfxi_replay_rejects_response_window_shaped_source_rows() -> None:
    source, labels, contexts, active = _fixture()
    response_window = source.drop(columns=list(VEC8_COLUMNS)).assign(
        **{column: 0.0 for column in ("r00", "r10", "r01", "r11", "b00", "b10", "b01", "b11")}
    )

    with pytest.raises(ValueError, match="explicit SFXI vec8 columns"):
        build_observed_sfxi_decomposition(
            response_window,
            labels,
            view_contexts=contexts,
            active_identities=active,
            top_k=2,
        )


def test_observed_sfxi_replay_rejects_label_or_denominator_drift() -> None:
    source, labels, contexts, active = _fixture()
    drifted_labels = labels.copy()
    drifted = np.asarray(drifted_labels.at[0, "y_obs"], dtype=float).copy()
    drifted[0] += 0.01
    drifted_labels.at[0, "y_obs"] = drifted

    with pytest.raises(ValueError, match="does not match the historical label ledger"):
        build_observed_sfxi_decomposition(
            source,
            drifted_labels,
            view_contexts=contexts,
            active_identities=active,
            top_k=2,
        )

    first, *rest = contexts
    drifted_context = ObservedSfxiViewContext(
        selection_view_id=first.selection_view_id,
        target_mask=first.target_mask,
        denom=first.denom + 0.01,
        scaling_percentile=first.scaling_percentile,
        scaling_min_n=first.scaling_min_n,
        scaling_eps=first.scaling_eps,
        intensity_log2_offset_delta=first.intensity_log2_offset_delta,
        source_campaign_slug=first.source_campaign_slug,
        source_run_id=first.source_run_id,
    )
    with pytest.raises(ValueError, match="persisted denominator"):
        build_observed_sfxi_decomposition(
            source,
            labels,
            view_contexts=(drifted_context, *rest),
            active_identities=active,
            top_k=2,
        )


def test_observed_sfxi_replay_rejects_promoted_identity_sequence_drift() -> None:
    source, labels, contexts, active = _fixture()
    drifted = active.copy()
    drifted.loc[0, "sequence"] = "SEQUENCE-DRIFT"

    with pytest.raises(ValueError, match="sequence does not match"):
        build_observed_sfxi_decomposition(
            source,
            labels,
            view_contexts=contexts,
            active_identities=drifted,
            top_k=2,
        )


def _fixture() -> tuple[pd.DataFrame, pd.DataFrame, tuple[ObservedSfxiViewContext, ...], pd.DataFrame]:
    records: list[dict[str, object]] = []
    for index in range(8):
        logic_shift = index / 14.0
        row: dict[str, object] = {
            "id": f"candidate-{index}",
            "sequence": f"ACGT-{index}",
            "design_id": (
                "pDual-10-spyp" if index == 6 else "pDual-10-sulAp" if index == 7 else f"pDual-10-ES{index + 1}p"
            ),
            "reader_experiment_id": f"experiment-{index % 3}",
            "v00": 0.10 + logic_shift,
            "v10": 0.75 - logic_shift / 2.0,
            "v01": 0.20 + logic_shift / 3.0,
            "v11": 0.90 - logic_shift / 4.0,
            "y00_star": -1.5 + index * 0.12,
            "y10_star": -0.8 + index * 0.18,
            "y01_star": -1.2 + index * 0.10,
            "y11_star": -0.4 + index * 0.22,
        }
        records.append(row)
    source = pd.DataFrame.from_records(records)
    labels = pd.DataFrame(
        {
            "id": source["id"],
            "sequence": source["sequence"],
            "observed_round": 0,
            "y_obs": [row.copy() for row in source.loc[:, VEC8_COLUMNS].to_numpy(dtype=float)],
        }
    )
    vec8 = source.loc[:, VEC8_COLUMNS].to_numpy(dtype=float)
    specs = (
        ("ethanol", (0.0, 1.0, 0.0, 1.0)),
        ("ciprofloxacin", (0.0, 0.0, 1.0, 1.0)),
        ("and", (0.0, 0.0, 0.0, 1.0)),
    )
    contexts = tuple(
        ObservedSfxiViewContext(
            selection_view_id=view_id,
            target_mask=mask,
            denom=float(score_vec8(vec8, SFXIScoringConfig(setpoint_vector=mask)).denom_used),
            scaling_percentile=95,
            scaling_min_n=5,
            scaling_eps=1.0e-8,
            intensity_log2_offset_delta=0.0,
            source_campaign_slug=f"source-{view_id}",
            source_run_id=f"run-{view_id}",
        )
        for view_id, mask in specs
    )
    active = source.loc[:4, ["id", "sequence"]].reset_index(drop=True)
    return source, labels, contexts, active
