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
import tomllib
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
    response_uncertainty,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.evaluation import (
    multistate_behavior_allocation as behavior_allocation,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.evaluation import (
    multistate_behavior_cardinality as behavior_cardinality,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.evaluation import (
    multistate_behavior_face_validity as behavior_face_validity,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.evaluation import (
    multistate_behavior_grouped_validation as behavior_grouped,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.evaluation import (
    multistate_behavior_normalization_sensitivity as behavior_sensitivity,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.evaluation import (
    multistate_behavior_rmf_replay as behavior_rmf,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.runtime import (
    multistate_behavior_audit_verification as behavior_audit_verification,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.runtime import (
    multistate_behavior_censor as behavior_censor,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.runtime import (
    multistate_behavior_completion as behavior_completion,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.runtime import (
    multistate_behavior_labels as behavior_labels,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.runtime import (
    multistate_behavior_publication as behavior_publication,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.runtime import (
    multistate_behavior_reference as behavior_reference,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.runtime import (
    multistate_behavior_shadow as behavior_runtime,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.runtime import (
    multistate_behavior_source_equivalence as behavior_equivalence,
)

PACKAGE = Path("src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy")
PROTOCOL_PATH = PACKAGE / "config/multistate_response_behavior_shadow_v1.yaml"
AUDIT_PATH = PACKAGE / "config/multistate_response_behavior_adversarial_audit_v1.json"


def test_adversarial_audit_is_declared_as_package_data() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    studies_package_data = pyproject["tool"]["setuptools"]["package-data"]["dnadesign.studies"]

    assert "units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/config/*.json" in studies_package_data
    assert "units/stress_ethanol_cipro_growth/decision/opal/multistate_response_behavior/*.yaml" in studies_package_data
    assert "units/stress_ethanol_cipro_growth/decision/opal/multistate_response_behavior/*.json" in studies_package_data


def test_adversarial_audit_pins_reviewed_snapshot_and_rejects_provenance_drift() -> None:
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))

    assert audit["auditor_id"] == "codex_subagent.metric_adversarial_audit.v1"
    assert audit["completed_at"] == "2026-07-17T22:49:20Z"
    assert audit["reviewed_source_commit"] == "ee83c807d88f2f58958dc8cff78290dd90dbb826"  # pragma: allowlist secret
    assert audit["reviewed_preliminary_manifest_sha256"] == (
        "sha256:dbf9e651a467df76f12ffa22c6b15f8515264a8d515731a601fc26274c66869f"
    )
    behavior_audit_verification.verify_behavior_adversarial_audit_record(audit)

    for field in (
        "auditor_id",
        "completed_at",
        "reviewed_source_commit",
        "reviewed_preliminary_manifest_sha256",
    ):
        drifted = dict(audit)
        drifted[field] = "drifted"
        with pytest.raises(ValueError, match="independent audit"):
            behavior_audit_verification.verify_behavior_adversarial_audit_record(drifted)


def test_checked_in_behavior_protocol_is_shadow_only_and_complete() -> None:
    protocol = load_multistate_behavior_protocol(PROTOCOL_PATH)

    assert protocol.schema_id == "stress_ethanol_cipro_growth.multistate_response_behavior_shadow.v1"
    assert protocol.protocol_id == "secg_multistate_response_behavior_shadow_v1"
    assert protocol.source_equivalence.prior_observation_bundle_repo_path.endswith(
        "workbench/outputs/response_window_observations/4_8h_v1"
    )
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


def test_source_observation_records_resolve_from_digest_bound_study_bundle(tmp_path: Path) -> None:
    protocol = load_multistate_behavior_protocol(PROTOCOL_PATH)
    dataset_root = tmp_path / "dataset"
    provenance_copy_root = dataset_root / "_opal/labels"
    source_root = tmp_path / "study-observations"
    provenance_copy_root.mkdir(parents=True)
    source_root.mkdir()
    observations = pd.DataFrame(
        {
            "candidate_id": ["candidate-a"],
            "display_label": ["Candidate A"],
            "label_source_reader_experiment_id": ["experiment-a"],
            **{
                component: [float(index)]
                for index, component in enumerate(
                    tuple(f"{prefix}{state}" for prefix in ("r", "b") for state in protocol.state_ids)
                )
            },
        }
    )
    observations_path = source_root / "observations.parquet"
    observations.to_parquet(observations_path, index=False)
    observation_manifest = {
        "observation_contract": {
            "primary_reduction_id": protocol.primary_reduction_id,
            "primary_value_requirement": "exact",
            "nonexact_label_action": "exclude_candidate",
            "y_space": "reader_response_window_vector_v1",
        },
        "records": {
            "observations": {
                "path": "observations.parquet",
                "sha256": hashlib.sha256(observations_path.read_bytes()).hexdigest(),
            }
        },
    }
    source_manifest_path = source_root / "manifest.json"
    source_manifest_path.write_text(json.dumps(observation_manifest, sort_keys=True), encoding="utf-8")
    manifest_sha = hashlib.sha256(source_manifest_path.read_bytes()).hexdigest()
    copied_manifest_path = provenance_copy_root / "source_observation.manifest.json"
    copied_manifest_path.write_bytes(source_manifest_path.read_bytes())
    provenance_path = provenance_copy_root / "study_provenance.json"
    provenance_path.write_text(
        json.dumps(
            {
                "observation_bundle": {
                    "manifest_path": "_opal/labels/source_observation.manifest.json",
                    "manifest_sha256": manifest_sha,
                }
            }
        ),
        encoding="utf-8",
    )

    observed_sha, loaded = behavior_labels._verify_exact_source_observation(
        dataset_root=dataset_root,
        study_provenance_path=provenance_path,
        source_observation_bundle_root=source_root,
        protocol=protocol,
    )

    assert observed_sha == manifest_sha
    pd.testing.assert_frame_equal(loaded, observations)
    assert not (provenance_copy_root / "observations.parquet").exists()


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


def test_shadow_protocol_cannot_authorize_a_checked_in_campaign() -> None:
    shadow = load_multistate_behavior_protocol(PROTOCOL_PATH)
    assert shadow.status == "shadow_only"
    assert shadow.campaign_activation == "prohibited"
    active_path = Path(
        "src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/"
        "multistate_response_behavior/protocol.yaml"
    )
    active = yaml.safe_load(active_path.read_text(encoding="utf-8"))
    assert active["status"] == "active_learning_probe"
    assert active["protocol_id"] != shadow.protocol_id

    objective_users: list[str] = []
    for path in sorted(Path("src/dnadesign/opal/campaigns").glob("*/configs/campaign.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if shadow.objective_name not in _nested_strings(payload):
            continue
        objective_users.append(str(path))
        metadata = payload["campaign"]["metadata"]
        assert metadata["protocol_id"] == active["protocol_id"]
        assert shadow.protocol_id not in _nested_strings(payload)

    assert objective_users == ["src/dnadesign/opal/campaigns/secg_msrb_greedy/configs/campaign.yaml"]


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
    assert result.signal_scale > 0.0
    assert result.bootstrap_samples == 120
    assert result.unit_count == 6
    assert result.response_pair_count == 6
    assert len(result.response_resolution_rows) == 36
    assert len(result.signal_resolution_rows) == 24
    assert result.response_scale == event_changed.response_scale
    assert result.signal_scale == event_changed.signal_scale
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
    experiment_by_id = {
        unit_id: f"experiment-{1 + index // 2}" for index, unit_id in enumerate(labels["id"].astype(str))
    }
    labels["reader_experiment_id"] = labels["id"].astype(str).map(experiment_by_id)
    labels["design_id"] = labels["candidate_id"]
    labels.loc[0, "design_id"] = "pDual-10-spyp"
    labels.loc[1, "design_id"] = "pDual-10-sulAp"
    labels["display_label"] = labels["candidate_id"]
    receipt = replace(
        _cohort_receipt(labels, draws, protocol=protocol),
        reader_experiment_count=3,
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
    predictions = _publication_prediction_rows(labels, count=24).assign(
        prediction_source_sha256=prediction_source_sha256
    )
    shadow = build_multistate_behavior_shadow_evidence(
        observed=labels,
        bootstrap_draws=draws,
        predictions=predictions,
        protocol=protocol,
        normalization=normalization,
        target_views=_target_views(),
    )
    rmf_uncertainty = response_uncertainty.estimate_response_calibration_from_reader_draws(
        labels,
        draws,
        target_views=_target_views(),
        scale_quantile=protocol.completion_gate.normalization_primary_quantile,
        expected_bootstrap_samples=normalization.bootstrap_samples,
    )
    hard_scores = behavior_rmf.build_current_rmf_prediction_scores(
        predictions=predictions,
        calibration=rmf_uncertainty.calibration,
        protocol=protocol,
        target_views=_target_views(),
    )
    comparison = compare_hard_and_behavior_scores(
        hard_scores,
        shadow.prediction_scores,
        top_k=6,
        hard_score_semantics="response_magnitude_feasibility_v1.feasibility_margin.maximize",
    )
    candidate_records = _publication_candidate_records(predictions)
    normalization_sensitivity = behavior_sensitivity.build_multistate_behavior_normalization_sensitivity(
        response_resolution_rows=normalization.response_resolution_rows,
        signal_resolution_rows=normalization.signal_resolution_rows,
        predictions=predictions,
        protocol=protocol,
        target_views=_target_views(),
        normalization_source_rows_sha256="sha256:" + normalization.source_rows_sha256,
        prediction_run_id="run-1",
        prediction_source_sha256=prediction_source_sha256,
    )
    validation_labels = _publication_validation_labels(labels)
    grouped = behavior_grouped.build_grouped_objective_validation(
        labels=validation_labels.labels,
        x=validation_labels.x,
        response_resolution_rows=normalization.response_resolution_rows,
        signal_resolution_rows=normalization.signal_resolution_rows,
        rmf_uncertainty_rows=rmf_uncertainty.rows,
        bootstrap_samples=normalization.bootstrap_samples,
        protocol=protocol,
        target_views=_target_views(),
        model_params=_registered_model_params(),
        source=validation_labels.source,
    )
    allocation = behavior_allocation.build_multistate_behavior_allocation_comparison(
        hard_behavior_detail=comparison.detail,
        candidate_records=candidate_records,
        protocol=protocol,
    )
    rmf_replay_calibration = behavior_rmf.bind_current_rmf_calibration(
        rmf_uncertainty.calibration,
        reader_bundle_manifest_sha256="1" * 64,
        normalization_source_rows_sha256=normalization.source_rows_sha256,
    )
    completion = behavior_completion.MultistateBehaviorCompletionEvidence(
        normalization_sensitivity=normalization_sensitivity,
        grouped_objective_validation=grouped,
        allocation_comparison=allocation,
        observed_control_face_validity=behavior_face_validity.build_behavior_face_validity(
            shadow.observed_scores,
            labels.drop(columns="id"),
            protocol=protocol,
        ),
        family_cardinality_pressure=behavior_cardinality.build_family_cardinality_pressure(protocol),
        validation_labels=validation_labels,
        rmf_resolution_rows=_publication_rmf_resolution(rmf_uncertainty.rows),
        rmf_replay_calibration=rmf_replay_calibration,
        prediction_vectors=_publication_prediction_vectors(predictions, candidate_records),
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
        completion=completion,
        reference_identity=behavior_reference.ReferenceSignalIdentityReceipt(
            reference_unit_count=3,
            bootstrap_row_count=360,
            reader_experiment_count=3,
        ),
        source={
            "prediction": {
                "run_id": "run-1",
                "ledger_root": "outputs/ledger",
                "ledger_sha256": prediction_source_sha256,
                "files": source_files,
                "candidate_count": 24,
                "run_receipt_scored_count": 24,
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
                    "source_row_count": 24,
                    "scored_row_count": 24,
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
    report = (bundle / "report.md").read_text(encoding="utf-8")
    assert "| Reader experiment | Candidate-experiment rank |" in report
    assert "| experiment-1 |" in report
    assert behavior_publication.verify_multistate_behavior_shadow(bundle) == manifest

    audit_tamper = tmp_path / "audit-tamper"
    shutil.copytree(bundle, audit_tamper)
    audit_manifest = json.loads((audit_tamper / "manifest.json").read_text(encoding="utf-8"))
    audit_path = audit_tamper / audit_manifest["artifacts"]["independent_adversarial_audit"]["path"]
    audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
    audit_payload["reviewed_source_commit"] = "f" * 40
    audit_path.write_text(json.dumps(audit_payload), encoding="utf-8")
    _refresh_artifact_receipt(audit_tamper, audit_manifest, artifact_id="independent_adversarial_audit")
    decision_path = audit_tamper / audit_manifest["artifacts"]["decision"]["path"]
    decision_payload = json.loads(decision_path.read_text(encoding="utf-8"))
    decision_payload["independent_adversarial_implementation_audit"]["reviewed_source_commit"] = "f" * 40
    decision_payload["independent_adversarial_implementation_audit"]["evidence_sha256"] = (
        "sha256:" + audit_manifest["artifacts"]["independent_adversarial_audit"]["sha256"]
    )
    decision_path.write_text(json.dumps(decision_payload), encoding="utf-8")
    _refresh_artifact_receipt(audit_tamper, audit_manifest, artifact_id="decision")
    (audit_tamper / "manifest.json").write_text(json.dumps(audit_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="reviewed snapshot"):
        behavior_publication.verify_multistate_behavior_shadow(audit_tamper)

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


def _publication_prediction_rows(labels: pd.DataFrame, *, count: int) -> pd.DataFrame:
    source = labels.loc[:, list(_component_columns())].to_numpy(dtype=float)
    records = []
    for index in range(count):
        values = source[index % len(source)] + 0.013 * index
        records.append(
            {
                "id": f"candidate-{index:02d}",
                **dict(zip(_component_columns(), values, strict=True)),
                "prediction_run_id": "run-1",
                "prediction_source_sha256": "sha256:" + "a" * 64,
            }
        )
    return pd.DataFrame.from_records(records)


def _publication_candidate_records(predictions: pd.DataFrame) -> pd.DataFrame:
    ids = predictions["id"].astype(str).tolist()
    return pd.DataFrame(
        {
            "id": ids,
            "sequence": [f"ACGT{index:024d}" for index in range(len(ids))],
            "usr_label__primary": [f"Candidate {index:02d}" for index in range(len(ids))],
        }
    )


def _publication_validation_labels(labels: pd.DataFrame) -> behavior_labels.VerifiedBehaviorValidationLabels:
    components = list(_component_columns())
    rows = labels.loc[:, ["candidate_id", "display_label", "reader_experiment_id", *components]].rename(
        columns={"reader_experiment_id": "label_source_reader_experiment_id"}
    )
    equivalence_rows = rows.loc[:, ["candidate_id", "label_source_reader_experiment_id"]].copy()
    equivalence_rows["observed_y"] = rows.loc[:, components].to_numpy(dtype=float).tolist()
    rng = np.random.default_rng(82)
    return behavior_labels.VerifiedBehaviorValidationLabels(
        labels=rows.reset_index(drop=True),
        x=rng.normal(size=(len(rows), 8)),
        source={
            "promotion_manifest_sha256": "sha256:" + "b" * 64,
            "candidate_records_sha256": "sha256:" + "c" * 64,
            "source_observation_manifest_sha256": "sha256:" + "d" * 64,
            "x_column_name": "test_x",
        },
        label_artifact_sha256="sha256:" + "e" * 64,
        central_label_equivalence_sha256=behavior_equivalence.grouped_central_equivalence_sha256(equivalence_rows),
        promoted_label_event_count=len(rows),
        promoted_candidate_count=len(rows),
    )


def _publication_rmf_resolution(frame: pd.DataFrame) -> pd.DataFrame:
    components = ("response_separation", "on_magnitude_floor", "off_magnitude_ceiling")
    return (
        frame.loc[
            :,
            [
                "id",
                "selection_view_id",
                "reader_experiment_id",
                *(f"{component}__combined_sd" for component in components),
            ],
        ]
        .sort_values(["selection_view_id", "id"], kind="mergesort")
        .reset_index(drop=True)
    )


def _publication_prediction_vectors(
    predictions: pd.DataFrame,
    candidate_records: pd.DataFrame,
) -> pd.DataFrame:
    metadata = candidate_records.copy()
    metadata["display_label"] = metadata["usr_label__primary"]
    metadata["sequence_sha256"] = metadata["sequence"].map(
        lambda value: hashlib.sha256(str(value).encode("ascii")).hexdigest()
    )
    rows = predictions.merge(
        metadata.loc[:, ["id", "display_label", "sequence_sha256"]],
        on="id",
        how="left",
        validate="one_to_one",
    )
    rows = rows.loc[
        :,
        [
            "id",
            "display_label",
            "sequence_sha256",
            "prediction_run_id",
            "prediction_source_sha256",
            *_component_columns(),
        ],
    ]
    rows["evidence_role"] = "fixed_raw_response_window_prediction_for_objective_replay"
    return rows


def _registered_model_params() -> dict[str, object]:
    return {
        "n_estimators": 100,
        "criterion": "friedman_mse",
        "bootstrap": True,
        "oob_score": True,
        "random_state": 7,
        "n_jobs": -1,
        "emit_feature_importance": True,
    }


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
