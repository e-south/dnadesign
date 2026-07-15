"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/response_metastudy/test_model_evidence_trajectory.py

Immutable model-evidence trajectory tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.model_evidence import (
    ModelEvidenceError,
    rebuild_catalog,
    record_checkpoint,
    verify_trajectory,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.model_evidence.cli import (
    main,
)


def test_checkpoint_record_is_immutable_and_idempotent(tmp_path: Path) -> None:
    bundle = _metastudy_bundle(tmp_path / "bundle")
    trajectory = tmp_path / "trajectory"

    first = record_checkpoint(
        metastudy_bundle=bundle,
        trajectory_root=trajectory,
        evidence_id="pre_batch0_retrospective",
    )
    second = record_checkpoint(
        metastudy_bundle=bundle,
        trajectory_root=trajectory,
        evidence_id="pre_batch0_retrospective",
    )

    assert second == first
    assert Path(first["checkpoint_path"]).is_file()
    assert len(list((trajectory / "series" / first["protocol_digest"] / "checkpoints").glob("*/checkpoint.json"))) == 1
    assert (
        json.loads((trajectory / "latest.json").read_text(encoding="utf-8"))["checkpoint_digest"]
        == first["checkpoint_digest"]
    )
    assert verify_trajectory(trajectory)["checkpoint_count"] == 1


def test_same_evidence_id_with_changed_content_fails_closed(tmp_path: Path) -> None:
    bundle = _metastudy_bundle(tmp_path / "bundle")
    trajectory = tmp_path / "trajectory"
    record_checkpoint(
        metastudy_bundle=bundle,
        trajectory_root=trajectory,
        evidence_id="pre_batch0_retrospective",
    )
    _rewrite_manifest(bundle, model_screen_candidate_count=36)

    with pytest.raises(ModelEvidenceError, match="already exists with different content"):
        record_checkpoint(
            metastudy_bundle=bundle,
            trajectory_root=trajectory,
            evidence_id="pre_batch0_retrospective",
        )


def test_protocol_change_starts_a_new_trajectory_series(tmp_path: Path) -> None:
    first_bundle = _metastudy_bundle(tmp_path / "bundle-a")
    second_bundle = _metastudy_bundle(tmp_path / "bundle-b", model_gate=0.35)
    trajectory = tmp_path / "trajectory"

    first = record_checkpoint(
        metastudy_bundle=first_bundle,
        trajectory_root=trajectory,
        evidence_id="pre_batch0_retrospective",
    )
    second = record_checkpoint(
        metastudy_bundle=second_bundle,
        trajectory_root=trajectory,
        evidence_id="pre_batch0_retrospective",
    )

    assert first["protocol_digest"] != second["protocol_digest"]
    assert len(list((trajectory / "series").iterdir())) == 2
    assert verify_trajectory(trajectory)["protocol_count"] == 2


def test_fitted_calibration_change_remains_a_result_in_one_protocol_series(tmp_path: Path) -> None:
    first_bundle = _metastudy_bundle(tmp_path / "bundle-a")
    second_bundle = _metastudy_bundle(tmp_path / "bundle-b")
    manifest_path = second_bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["response_metric_screen"]["review_calibration_by_selection_view"]["and"]["response_separation"][
        "scale"
    ] = 1.25
    manifest_path.write_text(json.dumps(manifest, allow_nan=False, sort_keys=True), encoding="utf-8")
    trajectory = tmp_path / "trajectory"

    first = record_checkpoint(
        metastudy_bundle=first_bundle,
        trajectory_root=trajectory,
        evidence_id="pre_batch0_retrospective",
    )
    second = record_checkpoint(
        metastudy_bundle=second_bundle,
        trajectory_root=trajectory,
        evidence_id="post_batch1_retrospective",
    )

    assert second["protocol_digest"] == first["protocol_digest"]
    assert second["checkpoint_digest"] != first["checkpoint_digest"]


def test_trajectory_verifier_rejects_immutable_checkpoint_tamper(tmp_path: Path) -> None:
    bundle = _metastudy_bundle(tmp_path / "bundle")
    trajectory = tmp_path / "trajectory"
    record = record_checkpoint(
        metastudy_bundle=bundle,
        trajectory_root=trajectory,
        evidence_id="pre_batch0_retrospective",
    )
    checkpoint_path = Path(record["checkpoint_path"])
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    payload["snapshot"]["corpus"]["model_screen_candidate_count"] = 999
    checkpoint_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ModelEvidenceError, match="checkpoint content digest mismatch"):
        verify_trajectory(trajectory)


def test_trajectory_verifier_rejects_frozen_protocol_tamper(tmp_path: Path) -> None:
    bundle = _metastudy_bundle(tmp_path / "bundle")
    trajectory = tmp_path / "trajectory"
    record = record_checkpoint(
        metastudy_bundle=bundle,
        trajectory_root=trajectory,
        evidence_id="pre_batch0_retrospective",
    )
    protocol_path = trajectory / "protocols" / record["protocol_digest"] / "protocol.json"
    payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    payload["protocol"]["model_support_gate"]["minimum_ordering_spearman"] = 0.01
    protocol_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ModelEvidenceError, match="protocol content digest mismatch"):
        verify_trajectory(trajectory)


def test_trajectory_verifier_rejects_unindexed_immutable_files(tmp_path: Path) -> None:
    bundle = _metastudy_bundle(tmp_path / "bundle")
    trajectory = tmp_path / "trajectory"
    record_checkpoint(
        metastudy_bundle=bundle,
        trajectory_root=trajectory,
        evidence_id="pre_batch0_retrospective",
    )
    (trajectory / "series" / "rogue.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ModelEvidenceError, match="unexpected file in immutable"):
        verify_trajectory(trajectory)


def test_catalog_is_rebuildable_and_not_scientific_authority(tmp_path: Path) -> None:
    bundle = _metastudy_bundle(tmp_path / "bundle")
    trajectory = tmp_path / "trajectory"
    record_checkpoint(metastudy_bundle=bundle, trajectory_root=trajectory, evidence_id="pre_batch0_retrospective")
    (trajectory / "catalog.json").write_text("{}", encoding="utf-8")

    catalog = rebuild_catalog(trajectory)

    assert catalog["checkpoint_count"] == 1
    indexed = catalog["checkpoints"][0]
    assert indexed["model_screen_candidate_count"] == 35
    assert indexed["campaign_model_summary"]["model_id"] == "campaign_random_forest"
    assert indexed["campaign_model_summary"]["median_channel_spearman"] == 0.2
    assert indexed["best_fixed_challenger_summary"]["model_id"] == "pls4"
    assert verify_trajectory(trajectory)["catalog_matches"] is True


def test_checkpoint_requires_explicit_label_and_decision_gates(tmp_path: Path) -> None:
    bundle = _metastudy_bundle(tmp_path / "bundle")
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["decision_gates"]["label_truth_ready"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ModelEvidenceError, match="decision_gates.label_truth_ready"):
        record_checkpoint(
            metastudy_bundle=bundle,
            trajectory_root=tmp_path / "trajectory",
            evidence_id="pre_batch0_retrospective",
        )


def test_checkpoint_requires_complete_campaign_model_evidence(tmp_path: Path) -> None:
    bundle = _metastudy_bundle(tmp_path / "bundle")
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["response_metric_screen"]["campaign_model_screen"]["validation"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ModelEvidenceError, match="campaign_model_screen missing fields.*validation"):
        record_checkpoint(
            metastudy_bundle=bundle,
            trajectory_root=tmp_path / "trajectory",
            evidence_id="pre_batch0_retrospective",
        )


def test_checkpoint_fails_closed_without_campaign_greedy_support(tmp_path: Path) -> None:
    bundle = _metastudy_bundle(tmp_path / "bundle")
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["response_metric_screen"]["campaign_greedy_support"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ModelEvidenceError, match="campaign_greedy_support must be a list"):
        record_checkpoint(
            metastudy_bundle=bundle,
            trajectory_root=tmp_path / "trajectory",
            evidence_id="pre_batch0_retrospective",
        )


def test_protocol_fingerprints_scientific_evaluators_without_ui_or_storage_sources(tmp_path: Path) -> None:
    bundle = _metastudy_bundle(tmp_path / "bundle")
    trajectory = tmp_path / "trajectory"

    record = record_checkpoint(
        metastudy_bundle=bundle,
        trajectory_root=trajectory,
        evidence_id="pre_batch0_retrospective",
    )

    protocol_path = trajectory / "protocols" / record["protocol_digest"] / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))["protocol"]
    paths = {row["path"] for row in protocol["evaluator_sources"]}
    assert any(path.endswith("evaluation/model_screen.py") for path in paths)
    assert any(path.endswith("runtime/response_screen.py") for path in paths)
    assert not any("/reporting/" in path for path in paths)
    assert not any("/model_evidence/" in path for path in paths)


def test_checkpoint_requires_the_complete_scientific_evaluator_fingerprint(tmp_path: Path) -> None:
    bundle = _metastudy_bundle(tmp_path / "bundle")
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source"]["files"] = [
        row for row in manifest["source"]["files"] if not row["path"].endswith("runtime/response_screen.py")
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ModelEvidenceError, match="missing required scientific evaluator.*runtime/response_screen.py"):
        record_checkpoint(
            metastudy_bundle=bundle,
            trajectory_root=tmp_path / "trajectory",
            evidence_id="pre_batch0_retrospective",
        )


def test_current_low_n_checkpoint_remains_retrospective_and_non_promoted(tmp_path: Path) -> None:
    bundle = _metastudy_bundle(tmp_path / "bundle")

    record = record_checkpoint(
        metastudy_bundle=bundle,
        trajectory_root=tmp_path / "trajectory",
        evidence_id="pre_batch0_retrospective",
    )
    checkpoint = json.loads(Path(record["checkpoint_path"]).read_text(encoding="utf-8"))
    snapshot = checkpoint["snapshot"]

    assert snapshot["evidence_timing"] == "retrospective"
    assert snapshot["decision_gates"] == {
        "label_truth_ready": False,
        "model_support_ready": False,
        "selection_policy_promoted": False,
        "synthesis_authorized": False,
    }
    assert snapshot["campaign_model"]["model_id"] == "campaign_random_forest"
    assert snapshot["campaign_model"]["median_channel_spearman"] == 0.2
    assert snapshot["campaign_model"]["minimum_channel_spearman"] == 0.0
    assert snapshot["campaign_model"]["response_magnitude_mae"] == 1.25
    assert snapshot["best_fixed_challenger"]["model_id"] == "pls4"
    assert snapshot["baseline"]["model_id"] == "mean_baseline"
    assert tuple(snapshot["per_view_evidence"]) == ("and", "ciprofloxacin", "ethanol")
    assert {
        snapshot["per_view_evidence"]["and"]["retrospective_campaign_model_greedy_support"]["model_id"],
        snapshot["per_view_evidence"]["and"]["retrospective_best_fixed_challenger_greedy_support"]["model_id"],
    } == {"campaign_random_forest", "pls4"}
    assert tuple(snapshot["review_calibration_by_selection_view"]) == ("and", "ciprofloxacin", "ethanol")
    assert snapshot["upstream_manifests"]["reader_response_window_bundle"]["sha256"] == "1" * 64
    assert "opal" not in snapshot


def test_cli_records_and_verifies_the_primary_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bundle = _metastudy_bundle(tmp_path / "bundle")
    trajectory = tmp_path / "trajectory"

    assert (
        main(
            [
                "record",
                "--metastudy-bundle",
                str(bundle),
                "--trajectory-root",
                str(trajectory),
                "--evidence-id",
                "pre_batch0_retrospective",
                "--json",
            ]
        )
        == 0
    )
    recorded = json.loads(capsys.readouterr().out)
    assert recorded["evidence_id"] == "pre_batch0_retrospective"

    assert main(["verify", "--trajectory-root", str(trajectory), "--json"]) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified["checkpoint_count"] == 1
    assert verified["catalog_matches"] is True


def _metastudy_bundle(root: Path, *, model_gate: float = 0.30) -> Path:
    root.mkdir(parents=True)
    artifact = root / "report.md"
    artifact.write_text("verified evidence", encoding="utf-8")
    manifest = _manifest(model_gate=model_gate)
    manifest["artifacts"] = {
        "report": {
            "path": "report.md",
            "bytes": artifact.stat().st_size,
            "sha256": _sha256(artifact.read_bytes()),
        }
    }
    (root / "manifest.json").write_text(json.dumps(manifest, allow_nan=False, sort_keys=True), encoding="utf-8")
    return root


def _rewrite_manifest(bundle: Path, *, model_screen_candidate_count: int) -> None:
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["response_metric_screen"]["model_screen_candidate_count"] = model_screen_candidate_count
    manifest_path.write_text(json.dumps(manifest, allow_nan=False, sort_keys=True), encoding="utf-8")


def _manifest(*, model_gate: float) -> dict[str, object]:
    views = ("and", "ciprofloxacin", "ethanol")
    campaign = _model_record("campaign_random_forest", "campaign_model", views, ordering=0.10)
    challenger = _model_record("pls4", "fixed_challenger", views, ordering=0.15)
    baseline = _model_record("mean_baseline", "baseline", views, ordering=0.0)
    return {
        "schema_version": "stress_ethanol_cipro_growth.response_metastudy.v10",
        "source": {
            "reader_bundle": {
                "manifest": {"path": "manifest.json", "sha256": "1" * 64, "bytes": 100},
                "counts": {"experiments": 8, "unique_design_ids": 40, "repeated_design_ids": 12},
            },
            "stress_campaign": {
                "config": {"path": "campaign.yaml", "sha256": "4" * 64, "bytes": 100},
            },
            "candidate_identity_binding": {
                "binding_count": 63,
                "candidate_count": 43,
                "files": [{"path": "manifest.json", "sha256": "2" * 64, "bytes": 100}],
            },
            "response_measurement_selection": {
                "scope": "model_screen_only",
                "label_truth_role": "none",
                "row_count": 35,
                "config": {
                    "path": (
                        "src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/"
                        "response_metastudy/config/response_model_screen_selection.yaml"
                    ),
                    "sha256": "3" * 64,
                    "bytes": 100,
                },
            },
            "target_views": [
                {"selection_view_id": "ethanol", "target_mask": [0, 1, 0, 1]},
                {"selection_view_id": "ciprofloxacin", "target_mask": [0, 0, 1, 1]},
                {"selection_view_id": "and", "target_mask": [0, 0, 0, 1]},
            ],
            "response_x_matrix_sha256": "5" * 64,
            "files": [
                {"path": path, "sha256": f"{index:x}" * 64, "bytes": 100}
                for index, path in enumerate(
                    (
                        "src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/"
                        "response_metastudy/core/response_contracts.py",
                        "src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/"
                        "response_metastudy/evaluation/greedy_support.py",
                        "src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/"
                        "response_metastudy/evaluation/grouped_models.py",
                        "src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/"
                        "response_metastudy/evaluation/model_screen.py",
                        "src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/"
                        "response_metastudy/evaluation/response_uncertainty.py",
                        "src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/"
                        "response_metastudy/runtime/response_screen.py",
                        "src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/"
                        "response_metastudy/reporting/response_model_plots.py",
                        "src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/"
                        "response_metastudy/model_evidence/storage.py",
                    ),
                    start=1,
                )
            ],
        },
        "label_truth": {
            "state": "not_ready",
            "source": "stress_ethanol_cipro_growth.response_window_observations",
            "screen_source_scope": "model_screen_only",
            "screen_source_label_truth_role": "none",
            "repeat_aggregation": "study_artifact_not_promoted",
            "observed_label_promotion_manifest": None,
        },
        "decision_gates": {
            "label_truth_ready": False,
            "model_support_ready": False,
            "selection_policy_promoted": False,
            "synthesis_authorized": False,
            "posture": "retrospective_screen_only",
            "opal_operational_state_included": False,
        },
        "response_metric_screen": {
            "status": "screen_complete_not_promoted",
            "evidence_timing": "retrospective",
            "model_screen_candidate_count": 35,
            "reader_event_experiment_count": 8,
            "repeated_design_count": 12,
            "maximum_screen_source_to_cross_experiment_median_abs_difference": 4.1,
            "model_support_basis": "configured_campaign_model",
            "model_support_ready": False,
            "campaign_model_screen": campaign,
            "best_fixed_model_screen": challenger,
            "baseline_model_screen": baseline,
            "prespecified_model_screens": [campaign, challenger, baseline],
            "fixed_model_definitions": [
                {"model_id": "mean_baseline", "kind": "mean", "role": "baseline", "target_transform": "none"},
                {
                    "model_id": "campaign_random_forest",
                    "kind": "random_forest",
                    "role": "campaign_model",
                    "target_transform": "none",
                },
                {"model_id": "pls4", "kind": "pls", "role": "fixed_challenger", "components": 4},
            ],
            "review_calibration_by_selection_view": {
                view: {
                    "response_separation": {"threshold": 0.0, "scale": 1.0},
                    "on_magnitude_floor": {"threshold": 0.0, "scale": 1.0},
                    "off_magnitude_ceiling": {"threshold": 0.0, "scale": 1.0},
                }
                for view in views
            },
            "response_screen_protocol": {
                "bootstrap_samples": 500,
                "scale_quantile": 0.9,
                "model_min_within_group_spearman": model_gate,
                "model_min_defined_group_count": 6,
                "model_reduction_ids": ["event_logmean_6_12h_post"],
                "reductions": [
                    {
                        "id": "event_logmean_6_12h_post",
                        "screen_role": "primary",
                        "response_basis": "post_window",
                        "method": "geometric_time_mean",
                        "window_start_event_h": 6.0,
                        "window_end_event_h": 12.0,
                    }
                ],
            },
            "campaign_greedy_support": [
                {
                    "selection_view_id": view,
                    "model_id": "campaign_random_forest",
                    "model_role": "campaign_model",
                    "evidence_basis": "configured_campaign_model",
                    "representation_id": "event_logmean_6_12h_post",
                    "held_out_group_count": 8,
                    "fraction_beating_group_median": 0.25,
                    "fraction_ci_low": 0.03,
                    "fraction_ci_high": 0.65,
                    "allocation_boundary": "descriptive_only_no_slot_assignment",
                }
                for view in views
            ],
            "best_fixed_challenger_greedy_support": [
                {
                    "selection_view_id": view,
                    "model_id": "pls4",
                    "model_role": "fixed_challenger",
                    "evidence_basis": "best_fixed_challenger",
                    "representation_id": "event_logmean_6_12h_post",
                    "held_out_group_count": 8,
                    "fraction_beating_group_median": 0.5,
                    "fraction_ci_low": 0.15,
                    "fraction_ci_high": 0.85,
                    "allocation_boundary": "descriptive_only_no_slot_assignment",
                }
                for view in views
            ],
        },
        "artifacts": {},
    }


def _model_record(
    model_id: str,
    role: str,
    views: tuple[str, ...],
    *,
    ordering: float,
) -> dict[str, object]:
    return {
        "model_id": model_id,
        "model_role": role,
        "representation_id": "event_logmean_6_12h_post",
        "target_transform": "none",
        "validation": "leave_one_reader_experiment_out",
        "metric_scope": "median_within_held_out_experiment",
        "weakest_target_view_response_separation_spearman": ordering,
        "weakest_target_view_feasibility_spearman": ordering,
        "weakest_required_ordering_spearman": ordering,
        "median_channel_spearman": ordering + 0.1,
        "minimum_channel_spearman": ordering - 0.1,
        "response_magnitude_mae": 1.25,
        "minimum_defined_group_count": 8,
        "target_view_ordering": {
            view: {
                "response_separation_spearman": ordering,
                "feasibility_spearman": ordering,
                "defined_group_count": 8,
            }
            for view in views
        },
        **({"configured_model_params": {"n_estimators": 100, "random_state": 7}} if role == "campaign_model" else {}),
    }


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
