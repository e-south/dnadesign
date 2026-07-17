"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/response_metastudy/test_multistate_behavior_shadow.py

Tests for the study-owned multistate behavior shadow protocol.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.core.contracts import (
    StressTargetView,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.evaluation import (
    BehaviorProtocolError,
    VerifiedBehaviorCohortReceipt,
    behavior_cohort_unit_ids_sha256,
    behavior_normalization_source_rows_sha256,
    bootstrap_rows_with_identity,
    build_bootstrap_rank_stability,
    build_multistate_behavior_normalization_record,
    build_multistate_behavior_shadow_evidence,
    compare_hard_and_behavior_scores,
    derive_multistate_behavior_normalization,
    load_multistate_behavior_protocol,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.runtime import (
    multistate_behavior_censor as behavior_censor,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.runtime import (
    multistate_behavior_publication as behavior_publication,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.runtime import (
    multistate_behavior_shadow as behavior_runtime,
)

PACKAGE = Path("src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy")
PROTOCOL_PATH = PACKAGE / "config/multistate_response_behavior_shadow_v1.yaml"


def test_checked_in_behavior_protocol_is_shadow_only_and_complete() -> None:
    protocol = load_multistate_behavior_protocol(PROTOCOL_PATH)

    assert protocol.schema_id == "stress_ethanol_cipro_growth.multistate_response_behavior_shadow.v1"
    assert protocol.protocol_id == "secg_multistate_response_behavior_shadow_v1"
    assert protocol.objective_name == "multistate_response_behavior_v1"
    assert protocol.status == "shadow_only"
    assert protocol.campaign_activation == "prohibited"
    assert protocol.synthesis_authorization == "prohibited"
    assert protocol.state_ids == ("00", "10", "01", "11")
    assert protocol.primary_reduction_id == "event_logmean_4_8h_post"
    assert protocol.normalization.normalized_temperature == 1.0
    assert protocol.normalization.scale_quantile == 0.9
    assert protocol.normalization.quantile_method == "linear"
    assert protocol.normalization.event_time_role == "separate_sensitivity_evidence"
    assert protocol.normalization.repeat_role == "separate_disagreement_evidence"
    assert protocol.normalization.censor_role == "exact_only_normalization_cohort"
    assert protocol.prediction_raw_top_k == 6
    assert protocol.comparison_role == "fixed_prediction_raw_candidate_ranking_no_sequence_allocation"
    assert protocol.target_masks == {
        "ethanol": (0.0, 1.0, 0.0, 1.0),
        "ciprofloxacin": (0.0, 0.0, 1.0, 1.0),
        "and": (0.0, 0.0, 0.0, 1.0),
    }
    assert len(protocol.source_sha256) == 64


def test_protocol_target_masks_must_match_runtime_views() -> None:
    protocol = load_multistate_behavior_protocol(PROTOCOL_PATH)
    drifted = (
        StressTargetView("ethanol", "Ethanol", (0.0, 1.0, 0.0, 1.0)),
        StressTargetView("ciprofloxacin", "Ciprofloxacin", (0.0, 0.0, 1.0, 1.0)),
        StressTargetView("and", "AND", (0.0, 1.0, 1.0, 1.0)),
    )

    with pytest.raises(BehaviorProtocolError, match="target masks disagree"):
        protocol.assert_target_views(drifted)


def test_protocol_rejects_state_identity_duplicate_yaml_and_numeric_strings(tmp_path: Path) -> None:
    payload = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    payload["assay"]["state_ids"] = ["a", "b", "c", "d"]
    drifted_states = tmp_path / "states.yaml"
    drifted_states.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(BehaviorProtocolError, match="state_ids must be exactly"):
        load_multistate_behavior_protocol(drifted_states)

    duplicated = tmp_path / "duplicate.yaml"
    duplicated.write_text(
        PROTOCOL_PATH.read_text(encoding="utf-8").replace(
            "status: shadow_only",
            "status: shadow_only\nstatus: shadow_only",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(BehaviorProtocolError, match="duplicate key"):
        load_multistate_behavior_protocol(duplicated)

    payload = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    payload["objective"]["normalized_temperature"] = "1.0"
    string_number = tmp_path / "string-number.yaml"
    string_number.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(BehaviorProtocolError, match="positive finite number"):
        load_multistate_behavior_protocol(string_number)


def test_shadow_protocol_cannot_be_named_by_a_checked_in_campaign() -> None:
    protocol = load_multistate_behavior_protocol(PROTOCOL_PATH)
    assert protocol.status == "shadow_only"
    assert protocol.campaign_activation == "prohibited"
    offenders: list[str] = []
    for path in sorted(Path("src/dnadesign/opal/campaigns").glob("*/configs/campaign.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if protocol.objective_name in _nested_strings(payload):
            offenders.append(str(path))
    assert offenders == []


def test_behavior_normalization_uses_bootstrap_only_and_one_common_scale_per_family() -> None:
    protocol = load_multistate_behavior_protocol(PROTOCOL_PATH)
    labels, draws = _reader_evidence(samples=120)
    target_views = _target_views()

    result = derive_multistate_behavior_normalization(
        labels,
        draws,
        protocol=protocol,
        target_views=target_views,
    )
    changed = labels.copy()
    event_columns = [column for column in changed if column.endswith("_event_half_range")]
    changed.loc[:, event_columns] = 10_000.0
    event_changed = derive_multistate_behavior_normalization(
        changed,
        draws,
        protocol=protocol,
        target_views=target_views,
    )

    assert result.response_scale > 0.0
    assert result.fluorescence_scale > 0.0
    assert result.bootstrap_samples == 120
    assert result.unit_count == 6
    assert result.response_pair_count == 6
    assert len(result.response_resolution_rows) == 36
    assert len(result.fluorescence_resolution_rows) == 24
    assert result.response_scale == event_changed.response_scale
    assert result.fluorescence_scale == event_changed.fluorescence_scale
    assert result.scale_basis == "reader_joint_bootstrap_component_resolution"
    assert result.event_time_role == "separate_sensitivity_evidence"
    assert result.repeat_role == "separate_disagreement_evidence"
    assert result.censor_role == "exact_only_normalization_cohort"
    assert len(result.source_rows_sha256) == 64


def test_behavior_normalization_deduplicates_pairs_declared_by_multiple_views() -> None:
    protocol = load_multistate_behavior_protocol(PROTOCOL_PATH)
    labels, draws = _reader_evidence(samples=120)

    result = derive_multistate_behavior_normalization(
        labels,
        draws,
        protocol=protocol,
        target_views=_target_views(),
    )

    pairs = set(result.response_resolution_rows[["state_a", "state_b"]].itertuples(index=False, name=None))
    assert pairs == {
        ("00", "10"),
        ("00", "01"),
        ("00", "11"),
        ("10", "01"),
        ("10", "11"),
        ("01", "11"),
    }
    assert result.response_resolution_rows.groupby(["id", "state_a", "state_b"]).size().max() == 1


def test_normalization_record_pins_protocol_cohort_sources_and_evidence_roles() -> None:
    protocol = load_multistate_behavior_protocol(PROTOCOL_PATH)
    labels, draws = _reader_evidence(samples=120)
    receipt = VerifiedBehaviorCohortReceipt(
        cohort_id="exact_primary_reader_candidate_experiments_v1",
        primary_reduction_id="event_logmean_4_8h_post",
        unit_count=6,
        candidate_count=6,
        reader_experiment_count=1,
        excluded_nonexact_unit_count=0,
        reader_bundle_manifest_sha256="1" * 64,
        candidate_bindings_manifest_sha256="3" * 64,
        unit_ids_sha256=behavior_cohort_unit_ids_sha256(labels),
        source_rows_sha256=behavior_normalization_source_rows_sha256(labels, draws, protocol=protocol),
    )
    result = derive_multistate_behavior_normalization(
        labels,
        draws,
        protocol=protocol,
        target_views=_target_views(),
        verified_cohort_receipt=receipt,
    )
    digests = {
        "reader_bundle_manifest_sha256": "1" * 64,
        "reader_request_sha256": "2" * 64,
        "candidate_bindings_manifest_sha256": "3" * 64,
        "observation_policy_sha256": "4" * 64,
    }

    record = build_multistate_behavior_normalization_record(result, source_artifact_digests=digests)

    assert record["status"] == "shadow_only"
    assert record["activation"] == {"campaign": "prohibited", "synthesis": "prohibited"}
    assert record["normalization"]["cohort_id"] == "exact_primary_reader_candidate_experiments_v1"
    assert record["normalization"]["response_pair_count"] == 6
    assert record["normalization"]["bootstrap_samples"] == 120
    assert record["evidence_roles"] == {
        "bootstrap": "normalization_and_candidate_experiment_unit_rank_sensitivity_no_top_k",
        "event_time": "separate_sensitivity_evidence",
        "repeat": "separate_disagreement_evidence",
        "censor": "exact_only_normalization_cohort",
    }
    assert record["source"]["reader_bundle_manifest_sha256"] == "sha256:" + "1" * 64
    assert record["source"]["protocol_sha256"] == "sha256:" + protocol.source_sha256


def test_behavior_normalization_rejects_nonexact_or_incomplete_evidence() -> None:
    protocol = load_multistate_behavior_protocol(PROTOCOL_PATH)
    labels, draws = _reader_evidence(samples=120)
    nonexact = labels.copy()
    nonexact.loc[0, "b01_bound_kind"] = "upper"

    with pytest.raises(ValueError, match="exact component evidence"):
        derive_multistate_behavior_normalization(
            nonexact,
            draws,
            protocol=protocol,
            target_views=_target_views(),
        )

    incomplete = draws.loc[~((draws["id"] == "candidate-b::experiment-1") & (draws["draw_index"] == 119))]
    with pytest.raises(ValueError, match="identical bootstrap draw counts"):
        derive_multistate_behavior_normalization(
            labels,
            incomplete,
            protocol=protocol,
            target_views=_target_views(),
        )


def test_bootstrap_identity_must_match_observed_rows() -> None:
    protocol = load_multistate_behavior_protocol(PROTOCOL_PATH)
    labels, draws = _reader_evidence(samples=120)
    conflicting = draws.assign(candidate_id="wrong-candidate", reader_experiment_id="wrong-experiment")

    with pytest.raises(ValueError, match="identity disagrees"):
        bootstrap_rows_with_identity(conflicting, labels, protocol=protocol)

    resolved = bootstrap_rows_with_identity(draws, labels, protocol=protocol)
    identity = labels.set_index("id")[["candidate_id", "reader_experiment_id"]]
    assert resolved.groupby("id")["candidate_id"].first().to_dict() == identity["candidate_id"].to_dict()
    assert (
        resolved.groupby("id")["reader_experiment_id"].first().to_dict() == identity["reader_experiment_id"].to_dict()
    )


def test_normalization_source_digest_distinguishes_one_ulp_float_drift() -> None:
    protocol = load_multistate_behavior_protocol(PROTOCOL_PATH)
    labels, draws = _reader_evidence(samples=120)
    baseline = behavior_normalization_source_rows_sha256(labels, draws, protocol=protocol)
    drifted = labels.copy()
    value = float(drifted.loc[0, "r00"])
    drifted.loc[0, "r00"] = np.nextafter(value, np.inf)

    assert behavior_normalization_source_rows_sha256(drifted, draws, protocol=protocol) != baseline


def test_shadow_builder_scores_observed_bootstrap_and_fixed_prediction_rows() -> None:
    protocol = load_multistate_behavior_protocol(PROTOCOL_PATH)
    labels, draws = _reader_evidence(samples=120)
    target_views = _target_views()
    normalization = derive_multistate_behavior_normalization(
        labels,
        draws,
        protocol=protocol,
        target_views=target_views,
    )
    predictions = _prediction_rows(labels)

    result = build_multistate_behavior_shadow_evidence(
        observed=labels,
        bootstrap_draws=draws,
        predictions=predictions,
        protocol=protocol,
        normalization=normalization,
        target_views=target_views,
    )

    assert len(result.observed_scores) == 18
    assert len(result.bootstrap_scores) == 2160
    assert len(result.prediction_scores) == 18
    assert set(result.observed_scores["selection_view_id"]) == {"ethanol", "ciprofloxacin", "and"}
    assert set(result.observed_scores["objective_name"]) == {"multistate_response_behavior_v1"}
    assert set(result.observed_scores["protocol_id"]) == {protocol.protocol_id}
    assert set(result.observed_scores["status"]) == {"shadow_only"}
    assert set(result.observed_scores["campaign_activation"]) == {"prohibited"}
    assert set(result.observed_scores["synthesis_authorization"]) == {"prohibited"}
    assert np.isfinite(result.observed_scores["behavior_score"]).all()
    assert np.isfinite(result.bootstrap_scores["behavior_score"]).all()
    assert np.isfinite(result.prediction_scores["behavior_score"]).all()
    grouped_weights = result.observed_coordinates.groupby(["id", "selection_view_id"])["bottleneck_weight"].sum()
    assert np.allclose(grouped_weights.to_numpy(dtype=float), 1.0)
    assert result.observed_coordinates["coordinate_label"].str.contains(":").all()
    assert len(result.event_sensitivity) == 18
    assert (
        result.event_sensitivity["behavior_score_worst_envelope"] <= result.event_sensitivity["behavior_score_central"]
    ).all()
    assert (
        result.event_sensitivity["behavior_score_central"] <= result.event_sensitivity["behavior_score_best_envelope"]
    ).all()
    assert set(result.event_sensitivity["event_bound_semantics"]) == {"componentwise_conservative_not_joint_event_draw"}
    assert set(result.event_sensitivity["event_bound_probability_claim"]) == {"none"}
    assert set(result.event_sensitivity["event_censor_posture"]) == {"exact_unclipped_unoverflowed"}
    assert "central_unit_rank" in result.event_sensitivity
    assert "central_rank" not in result.event_sensitivity
    assert len(result.bootstrap_rank_stability) == 3
    assert set(result.bootstrap_rank_stability["candidate_experiment_unit_count"]) == {6}
    assert "raw_top_k" not in result.bootstrap_rank_stability
    assert "raw_top_k_overlap_min" not in result.bootstrap_rank_stability
    assert set(result.bootstrap_rank_stability["bootstrap_draw_count"]) == {120}
    assert result.repeated_candidate_agreement.empty


def test_shadow_builder_rejects_normalization_source_drift_and_bad_event_ranges() -> None:
    protocol = load_multistate_behavior_protocol(PROTOCOL_PATH)
    labels, draws = _reader_evidence(samples=120)
    target_views = _target_views()
    normalization = derive_multistate_behavior_normalization(
        labels,
        draws,
        protocol=protocol,
        target_views=target_views,
    )
    predictions = _prediction_rows(labels)
    drifted = labels.copy()
    drifted.loc[0, "r10"] = float(drifted.loc[0, "r10"]) + 0.01

    with pytest.raises(ValueError, match="does not reproduce the normalization source rows"):
        build_multistate_behavior_shadow_evidence(
            observed=drifted,
            bootstrap_draws=draws,
            predictions=predictions,
            protocol=protocol,
            normalization=normalization,
            target_views=target_views,
        )

    negative_range = labels.copy()
    negative_range.loc[0, "b11_event_half_range"] = -0.1
    matching_normalization = derive_multistate_behavior_normalization(
        negative_range,
        draws,
        protocol=protocol,
        target_views=target_views,
    )
    with pytest.raises(ValueError, match="finite and nonnegative"):
        build_multistate_behavior_shadow_evidence(
            observed=negative_range,
            bootstrap_draws=draws,
            predictions=predictions,
            protocol=protocol,
            normalization=matching_normalization,
            target_views=target_views,
        )

    overflow = labels.copy()
    overflow.loc[0, "r11_event_sensitivity_has_instrument_overflow"] = True
    overflow_normalization = derive_multistate_behavior_normalization(
        overflow,
        draws,
        protocol=protocol,
        target_views=target_views,
    )
    with pytest.raises(ValueError, match="without instrument overflow"):
        build_multistate_behavior_shadow_evidence(
            observed=overflow,
            bootstrap_draws=draws,
            predictions=predictions,
            protocol=protocol,
            normalization=overflow_normalization,
            target_views=target_views,
        )


def test_shadow_builder_reports_repeated_candidate_disagreement_without_aggregation() -> None:
    protocol = load_multistate_behavior_protocol(PROTOCOL_PATH)
    labels, draws = _reader_evidence(samples=120)
    old_id = "candidate-b::experiment-1"
    new_id = "candidate-a::experiment-2"
    labels.loc[1, "id"] = new_id
    labels.loc[1, "candidate_id"] = "candidate-a"
    labels.loc[1, "reader_experiment_id"] = "experiment-2"
    draws.loc[draws["id"].eq(old_id), "id"] = new_id
    target_views = _target_views()
    normalization = derive_multistate_behavior_normalization(
        labels,
        draws,
        protocol=protocol,
        target_views=target_views,
    )
    predictions = _prediction_rows(labels.drop_duplicates("candidate_id"))

    result = build_multistate_behavior_shadow_evidence(
        observed=labels,
        bootstrap_draws=draws,
        predictions=predictions,
        protocol=protocol,
        normalization=normalization,
        target_views=target_views,
    )

    assert len(result.repeated_candidate_agreement) == 3
    assert set(result.repeated_candidate_agreement["candidate_id"]) == {"candidate-a"}
    assert set(result.repeated_candidate_agreement["experiment_count"]) == {2}
    assert (result.repeated_candidate_agreement["behavior_score_range"] >= 0.0).all()
    assert set(result.repeated_candidate_agreement["evidence_role"]) == {
        "repeat_agreement_only_no_label_aggregation_or_source_choice"
    }


def test_hard_vs_behavior_comparison_requires_aligned_ids_and_reports_topk() -> None:
    protocol = load_multistate_behavior_protocol(PROTOCOL_PATH)
    labels, draws = _reader_evidence(samples=120)
    target_views = _target_views()
    normalization = derive_multistate_behavior_normalization(
        labels,
        draws,
        protocol=protocol,
        target_views=target_views,
    )
    predictions = _prediction_rows(labels)
    shadow = build_multistate_behavior_shadow_evidence(
        observed=labels,
        bootstrap_draws=draws,
        predictions=predictions,
        protocol=protocol,
        normalization=normalization,
        target_views=target_views,
    )
    hard = shadow.prediction_scores.loc[
        :,
        [
            "id",
            "selection_view_id",
            "hard_bottleneck_clearance",
            "prediction_run_id",
            "prediction_source_sha256",
        ],
    ].rename(columns={"hard_bottleneck_clearance": "hard_score"})

    comparison = compare_hard_and_behavior_scores(
        hard,
        shadow.prediction_scores,
        top_k=1,
        hard_score_semantics="unthresholded_hard_bottleneck_clearance",
    )

    assert len(comparison.summary) == 3
    assert set(comparison.summary["raw_top_k"]) == {1}
    assert set(comparison.summary["hard_score_semantics"]) == {"unthresholded_hard_bottleneck_clearance"}
    assert set(comparison.summary["ranking_method"]) == {"descending_score_then_ascending_candidate_id"}
    assert set(comparison.summary["tie_semantics"]) == {"ordinal_rank_with_id_tiebreak"}
    assert (comparison.summary["raw_top_k_overlap"] >= 0).all()
    assert set(comparison.detail["behavior_rank"]) == {1, 2, 3, 4, 5, 6}
    assert comparison.detail.groupby("selection_view_id")["behavior_selected"].sum().eq(1).all()

    undefined = compare_hard_and_behavior_scores(
        hard.assign(hard_score=0.0),
        shadow.prediction_scores,
        top_k=1,
        hard_score_semantics="constant_test_comparator",
    )
    assert undefined.summary["hard_behavior_spearman"].isna().all()

    missing = hard.loc[~((hard["selection_view_id"] == "ethanol") & (hard["id"] == "candidate-a"))]
    with pytest.raises(ValueError, match="candidate ids disagree"):
        compare_hard_and_behavior_scores(
            missing,
            shadow.prediction_scores,
            top_k=1,
            hard_score_semantics="unthresholded_hard_bottleneck_clearance",
        )


def test_shadow_publication_is_atomic_complete_and_digest_verified(tmp_path: Path) -> None:
    protocol = load_multistate_behavior_protocol(PROTOCOL_PATH)
    labels, draws = _reader_evidence(samples=120)
    receipt = replace(
        _cohort_receipt(labels, draws, protocol=protocol),
        excluded_nonexact_unit_count=1,
    )
    normalization = derive_multistate_behavior_normalization(
        labels,
        draws,
        protocol=protocol,
        target_views=_target_views(),
        verified_cohort_receipt=receipt,
    )
    source_digests = {
        "reader_bundle_manifest_sha256": "1" * 64,
        "reader_request_sha256": "2" * 64,
        "candidate_bindings_manifest_sha256": "3" * 64,
        "observation_policy_sha256": "4" * 64,
    }
    normalization_record = build_multistate_behavior_normalization_record(
        normalization,
        source_artifact_digests=source_digests,
    )
    source_files = {
        "prediction_parts": [
            {
                "path": "outputs/ledger/predictions/part.parquet",
                "bytes": 10,
                "sha256": "5" * 64,
            }
        ],
        "run_receipt_parts": [
            {
                "path": "outputs/ledger/runs.parquet/part.parquet",
                "bytes": 11,
                "sha256": "6" * 64,
            }
        ],
    }
    prediction_source_sha256 = (
        "sha256:"
        + hashlib.sha256(json.dumps(source_files, separators=(",", ":"), sort_keys=True).encode("utf-8")).hexdigest()
    )
    predictions = _prediction_rows(labels).assign(prediction_source_sha256=prediction_source_sha256)
    shadow = build_multistate_behavior_shadow_evidence(
        observed=labels,
        bootstrap_draws=draws,
        predictions=predictions,
        protocol=protocol,
        normalization=normalization,
        target_views=_target_views(),
    )
    hard_scores = shadow.prediction_scores.loc[
        :,
        [
            "id",
            "selection_view_id",
            "hard_bottleneck_clearance",
            "prediction_run_id",
            "prediction_source_sha256",
        ],
    ].rename(columns={"hard_bottleneck_clearance": "hard_score"})
    comparison = compare_hard_and_behavior_scores(
        hard_scores,
        shadow.prediction_scores,
        top_k=6,
        hard_score_semantics="response_magnitude_feasibility_v1.feasibility_margin.maximize",
    )
    measurements = _censor_measurements(labels)
    excluded = measurements.iloc[[0]].copy()
    excluded["candidate_id"] = "excluded-candidate"
    excluded["design_id"] = "excluded-design"
    excluded["reader_experiment_id"] = "experiment-2"
    excluded["r00_bound_kind"] = "lower"
    excluded["r00_has_policy_clipping"] = True
    measurements = pd.concat([measurements, excluded], ignore_index=True)
    censor_exclusions = behavior_censor.build_behavior_censor_exclusions(
        measurements,
        primary_reduction_id=protocol.primary_reduction_id,
        state_ids=protocol.state_ids,
    )
    preview = behavior_runtime.VerifiedMultistateBehaviorShadow(
        normalization=normalization,
        normalization_record=normalization_record,
        evidence=shadow,
        hard_comparison=comparison,
        censor_exclusions=censor_exclusions,
        source={
            "prediction": {
                "run_id": "run-1",
                "ledger_root": "outputs/ledger",
                "ledger_sha256": prediction_source_sha256,
                "files": source_files,
                "candidate_count": 6,
                "run_receipt_scored_count": 6,
                "run_lineage": {
                    "as_of_round": 0,
                    "model_name": "random_forest",
                    "model_params_sha256": "sha256:" + "7" * 64,
                    "y_ingest_name": "vector_from_table_v1",
                    "y_ingest_params_sha256": "sha256:" + "8" * 64,
                    "training_y_ops_sha256": "sha256:" + "9" * 64,
                    "training_row_count": 6,
                },
                "candidate_projection": {
                    "source_row_count": 7,
                    "scored_row_count": 6,
                    "sha256": "sha256:" + "a" * 64,
                },
            },
            **{key: "sha256:" + value for key, value in source_digests.items()},
        },
    )
    bundle = tmp_path / "shadow"

    manifest = behavior_publication.publish_multistate_behavior_shadow(
        preview,
        out_dir=bundle,
        overwrite=False,
    )

    assert manifest["status"] == "shadow_only"
    assert manifest["activation"] == {"campaign": "prohibited", "synthesis": "prohibited"}
    assert manifest["tables"]["bootstrap_scores"]["rows"] == 2160
    assert manifest["tables"]["normalization_response_resolution"]["rows"] == 36
    assert behavior_publication.verify_multistate_behavior_shadow(bundle) == manifest

    claim_tamper = tmp_path / "claim-tamper"
    shutil.copytree(bundle, claim_tamper)
    claim_manifest = json.loads((claim_tamper / "manifest.json").read_text(encoding="utf-8"))
    claim_manifest["claim_boundary"] = "synthesis_authorized"
    (claim_tamper / "manifest.json").write_text(json.dumps(claim_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="claim_boundary"):
        behavior_publication.verify_multistate_behavior_shadow(claim_tamper)

    scale_tamper = tmp_path / "scale-tamper"
    shutil.copytree(bundle, scale_tamper)
    scale_manifest = json.loads((scale_tamper / "manifest.json").read_text(encoding="utf-8"))
    normalization_path = scale_tamper / scale_manifest["artifacts"]["normalization"]["path"]
    normalization_payload = json.loads(normalization_path.read_text(encoding="utf-8"))
    normalization_payload["normalization"]["response_scale"] = 999.0
    normalization_path.write_text(json.dumps(normalization_payload), encoding="utf-8")
    _refresh_artifact_receipt(scale_tamper, scale_manifest, artifact_id="normalization")
    for table_id in (
        "observed_scores",
        "observed_coordinates",
        "bootstrap_scores",
        "event_sensitivity",
        "prediction_scores",
    ):
        path = scale_tamper / scale_manifest["artifacts"][f"table__{table_id}"]["path"]
        frame = pd.read_parquet(path)
        frame["response_scale"] = 999.0
        frame.to_parquet(path, index=False)
        _refresh_artifact_receipt(scale_tamper, scale_manifest, artifact_id=f"table__{table_id}")
    (scale_tamper / "manifest.json").write_text(json.dumps(scale_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="does not derive"):
        behavior_publication.verify_multistate_behavior_shadow(scale_tamper)

    view_tamper = tmp_path / "view-tamper"
    shutil.copytree(bundle, view_tamper)
    view_manifest = json.loads((view_tamper / "manifest.json").read_text(encoding="utf-8"))
    observed_path = view_tamper / view_manifest["artifacts"]["table__observed_scores"]["path"]
    observed_frame = pd.read_parquet(observed_path)
    observed_frame["selection_view_id"] = observed_frame["selection_view_id"].map(lambda value: f"bogus-{value}")
    observed_frame.to_parquet(observed_path, index=False)
    _refresh_artifact_receipt(view_tamper, view_manifest, artifact_id="table__observed_scores")
    (view_tamper / "manifest.json").write_text(json.dumps(view_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="selection view"):
        behavior_publication.verify_multistate_behavior_shadow(view_tamper)

    identity_tamper = tmp_path / "identity-tamper"
    shutil.copytree(bundle, identity_tamper)
    identity_manifest = json.loads((identity_tamper / "manifest.json").read_text(encoding="utf-8"))
    coordinates_path = identity_tamper / identity_manifest["artifacts"]["table__observed_coordinates"]["path"]
    coordinates = pd.read_parquet(coordinates_path)
    coordinates.loc[0, "candidate_id"] = "wrong-candidate"
    coordinates.to_parquet(coordinates_path, index=False)
    _refresh_artifact_receipt(identity_tamper, identity_manifest, artifact_id="table__observed_coordinates")
    (identity_tamper / "manifest.json").write_text(json.dumps(identity_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="identity disagrees"):
        behavior_publication.verify_multistate_behavior_shadow(identity_tamper)

    censor_tamper = tmp_path / "censor-tamper"
    shutil.copytree(bundle, censor_tamper)
    censor_manifest = json.loads((censor_tamper / "manifest.json").read_text(encoding="utf-8"))
    censor_path = censor_tamper / censor_manifest["artifacts"]["table__censor_exclusions"]["path"]
    censor_rows = pd.read_parquet(censor_path)
    censor_rows["candidate_id"] = str(labels.loc[0, "candidate_id"])
    censor_rows["reader_experiment_id"] = str(labels.loc[0, "reader_experiment_id"])
    censor_rows.to_parquet(censor_path, index=False)
    _refresh_artifact_receipt(censor_tamper, censor_manifest, artifact_id="table__censor_exclusions")
    (censor_tamper / "manifest.json").write_text(json.dumps(censor_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="overlap exact observed units"):
        behavior_publication.verify_multistate_behavior_shadow(censor_tamper)

    event_rank_tamper = tmp_path / "event-rank-tamper"
    shutil.copytree(bundle, event_rank_tamper)
    event_rank_manifest = json.loads((event_rank_tamper / "manifest.json").read_text(encoding="utf-8"))
    event_rank_path = event_rank_tamper / event_rank_manifest["artifacts"]["table__event_sensitivity"]["path"]
    event_rank_rows = pd.read_parquet(event_rank_path)
    event_rank_rows.loc[:, "central_unit_rank"] = 1
    event_rank_rows.loc[:, "worst_envelope_unit_rank"] = 1
    event_rank_rows.loc[:, "best_envelope_unit_rank"] = 1
    event_rank_rows.loc[:, "event_unit_rank_min"] = 1
    event_rank_rows.loc[:, "event_unit_rank_max"] = 1
    event_rank_rows.loc[:, "event_unit_rank_span"] = 0
    event_rank_rows.to_parquet(event_rank_path, index=False)
    _refresh_artifact_receipt(event_rank_tamper, event_rank_manifest, artifact_id="table__event_sensitivity")
    (event_rank_tamper / "manifest.json").write_text(json.dumps(event_rank_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="does not derive"):
        behavior_publication.verify_multistate_behavior_shadow(event_rank_tamper)

    event_bottleneck_tamper = tmp_path / "event-bottleneck-tamper"
    shutil.copytree(bundle, event_bottleneck_tamper)
    event_bottleneck_manifest = json.loads((event_bottleneck_tamper / "manifest.json").read_text(encoding="utf-8"))
    event_bottleneck_path = (
        event_bottleneck_tamper / event_bottleneck_manifest["artifacts"]["table__event_sensitivity"]["path"]
    )
    event_bottleneck_rows = pd.read_parquet(event_bottleneck_path)
    event_bottleneck_rows.loc[0, "hard_bottleneck_worst_envelope"] = (
        float(event_bottleneck_rows.loc[0, "hard_bottleneck_best_envelope"]) + 1.0
    )
    event_bottleneck_rows.to_parquet(event_bottleneck_path, index=False)
    _refresh_artifact_receipt(
        event_bottleneck_tamper,
        event_bottleneck_manifest,
        artifact_id="table__event_sensitivity",
    )
    (event_bottleneck_tamper / "manifest.json").write_text(
        json.dumps(event_bottleneck_manifest),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="directionally inconsistent"):
        behavior_publication.verify_multistate_behavior_shadow(event_bottleneck_tamper)

    manifest_path = bundle / "manifest.json"
    manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest_path.write_text(
        manifest_text.replace(
            '"status": "shadow_only",',
            '"status": "shadow_only",\n  "status": "shadow_only",',
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate key"):
        behavior_publication.verify_multistate_behavior_shadow(bundle)
    manifest_path.write_text(manifest_text, encoding="utf-8")
    drifted_manifest = json.loads(manifest_text)
    drifted_manifest["source"]["prediction"]["files"]["prediction_parts"][0]["path"] = "/absolute/path"
    manifest_path.write_text(json.dumps(drifted_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="path escapes"):
        behavior_publication.verify_multistate_behavior_shadow(bundle)
    manifest_path.write_text(manifest_text, encoding="utf-8")
    prediction_path = bundle / manifest["artifacts"]["table__prediction_scores"]["path"]
    prediction_path.write_bytes(prediction_path.read_bytes() + b"drift")
    with pytest.raises(ValueError, match="size or digest mismatch"):
        behavior_publication.verify_multistate_behavior_shadow(bundle)


def test_censor_review_rejects_text_booleans() -> None:
    labels, _draws = _reader_evidence(samples=120)
    measurements = _censor_measurements(labels)
    measurements.loc[0, "r00_bound_kind"] = "upper"
    flag_column = "r00_has_instrument_overflow"
    measurements[flag_column] = measurements[flag_column].astype(object)
    measurements.loc[0, flag_column] = "False"

    with pytest.raises(ValueError, match="exact boolean"):
        behavior_censor.build_behavior_censor_exclusions(
            measurements,
            primary_reduction_id="event_logmean_4_8h_post",
            state_ids=("00", "10", "01", "11"),
        )


def test_censor_review_preserves_policy_clipping_and_rejects_inconsistent_bounds() -> None:
    labels, _draws = _reader_evidence(samples=120)
    measurements = _censor_measurements(labels)
    measurements.loc[0, "r00_bound_kind"] = "lower"
    measurements.loc[0, "r00_has_policy_clipping"] = True

    exclusions = behavior_censor.build_behavior_censor_exclusions(
        measurements,
        primary_reduction_id="event_logmean_4_8h_post",
        state_ids=("00", "10", "01", "11"),
    )

    row = exclusions.loc[exclusions["component"].eq("r00")].iloc[0]
    assert bool(row["has_policy_clipping"])
    assert not bool(row["has_instrument_overflow"])

    inconsistent = measurements.copy()
    inconsistent.loc[0, "r00_has_policy_clipping"] = False
    with pytest.raises(ValueError, match="bound kind disagrees"):
        behavior_censor.build_behavior_censor_exclusions(
            inconsistent,
            primary_reduction_id="event_logmean_4_8h_post",
            state_ids=("00", "10", "01", "11"),
        )


def test_bootstrap_rank_stability_reports_undefined_correlations_explicitly() -> None:
    ids = [f"candidate-{index}" for index in range(6)]
    observed = pd.DataFrame(
        {
            "id": ids,
            "selection_view_id": "and",
            "behavior_score": 0.0,
            "protocol_id": "protocol-1",
        }
    )
    draws = pd.DataFrame.from_records(
        [
            {
                "id": candidate_id,
                "selection_view_id": "and",
                "draw_index": draw_index,
                "behavior_score": float(index + draw_index),
                "protocol_id": "protocol-1",
            }
            for draw_index in range(2)
            for index, candidate_id in enumerate(ids)
        ]
    )

    result = build_bootstrap_rank_stability(observed, draws)

    row = result.summary.iloc[0]
    assert row["correlation_defined_draw_count"] == 0
    assert row["correlation_undefined_draw_count"] == 2
    assert pd.isna(row["central_draw_spearman_median"])
    assert result.draws["correlation_defined"].eq(False).all()


def _target_views() -> tuple[StressTargetView, ...]:
    return (
        StressTargetView("ethanol", "Ethanol", (0.0, 1.0, 0.0, 1.0)),
        StressTargetView("ciprofloxacin", "Ciprofloxacin", (0.0, 0.0, 1.0, 1.0)),
        StressTargetView("and", "AND", (0.0, 0.0, 0.0, 1.0)),
    )


def _component_columns() -> tuple[str, ...]:
    return ("r00", "r10", "r01", "r11", "b00", "b10", "b01", "b11")


def _prediction_rows(labels: pd.DataFrame) -> pd.DataFrame:
    return (
        labels.loc[:, ["candidate_id", *_component_columns()]]
        .rename(columns={"candidate_id": "id"})
        .assign(
            prediction_run_id="run-1",
            prediction_source_sha256="sha256:" + "a" * 64,
        )
    )


def _cohort_receipt(
    labels: pd.DataFrame,
    draws: pd.DataFrame,
    *,
    protocol,
) -> VerifiedBehaviorCohortReceipt:
    return VerifiedBehaviorCohortReceipt(
        cohort_id="exact_primary_reader_candidate_experiments_v1",
        primary_reduction_id="event_logmean_4_8h_post",
        unit_count=6,
        candidate_count=6,
        reader_experiment_count=1,
        excluded_nonexact_unit_count=0,
        reader_bundle_manifest_sha256="1" * 64,
        candidate_bindings_manifest_sha256="3" * 64,
        unit_ids_sha256=behavior_cohort_unit_ids_sha256(labels),
        source_rows_sha256=behavior_normalization_source_rows_sha256(
            labels,
            draws,
            protocol=protocol,
        ),
    )


def _censor_measurements(labels: pd.DataFrame) -> pd.DataFrame:
    result = labels.copy()
    result["design_id"] = result["candidate_id"]
    for component in _component_columns():
        result[f"{component}_has_policy_clipping"] = False
        result[f"{component}_has_instrument_overflow"] = False
    return result


def _refresh_artifact_receipt(bundle: Path, manifest: dict[str, object], *, artifact_id: str) -> None:
    record = manifest["artifacts"][artifact_id]
    path = bundle / record["path"]
    payload = path.read_bytes()
    record["bytes"] = len(payload)
    record["sha256"] = hashlib.sha256(payload).hexdigest()


def _nested_strings(value: object) -> set[str]:
    if isinstance(value, dict):
        return {item for nested in value.values() for item in _nested_strings(nested)}
    if isinstance(value, list):
        return {item for nested in value for item in _nested_strings(nested)}
    return {value} if isinstance(value, str) else set()


def _reader_evidence(*, samples: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    component_columns = _component_columns()
    label_rows: list[dict[str, object]] = []
    draw_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(17)
    for offset, candidate_id in enumerate(
        ("candidate-a", "candidate-b", "candidate-c", "candidate-d", "candidate-e", "candidate-f")
    ):
        unit_id = f"{candidate_id}::experiment-1"
        label: dict[str, object] = {
            "id": unit_id,
            "candidate_id": candidate_id,
            "reader_experiment_id": "experiment-1",
            "reduction_id": "event_logmean_4_8h_post",
        }
        for index, column in enumerate(component_columns):
            label[column] = float(index + offset)
            label[f"{column}_bound_kind"] = "exact"
            label[f"{column}_event_half_range"] = 0.1
            label[f"{column}_event_sensitivity_has_policy_clipping"] = False
            label[f"{column}_event_sensitivity_has_instrument_overflow"] = False
        label_rows.append(label)
        for draw_index in range(samples):
            draw: dict[str, object] = {"id": unit_id, "draw_index": draw_index}
            common = rng.normal(0.0, 0.02)
            for index, column in enumerate(component_columns):
                draw[column] = float(label[column]) + common + rng.normal(0.0, 0.02 + index * 0.003)
            draw_rows.append(draw)
    return pd.DataFrame.from_records(label_rows), pd.DataFrame.from_records(draw_rows)
