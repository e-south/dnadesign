"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/runtime/multistate_behavior_publication.py

Atomic publication and verification for multistate behavior shadow evidence.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..reporting.multistate_behavior_plots import render_multistate_behavior_plots
from ..reporting.multistate_behavior_report import render_multistate_behavior_report
from .multistate_behavior_audit_verification import verify_behavior_adversarial_audit_record
from .multistate_behavior_bundle_verification import (
    SCHEMA_ID,
    verify_multistate_behavior_shadow,
)
from .multistate_behavior_decision import build_multistate_behavior_decision
from .multistate_behavior_json import load_strict_behavior_json
from .multistate_behavior_prediction_surface_diagnostics import build_prediction_surface_diagnostics
from .multistate_behavior_shadow import VerifiedMultistateBehaviorShadow
from .multistate_behavior_source_equivalence import build_source_equivalence_receipt
from .publication import (
    artifact_inventory,
    create_staging_dir,
    publish_staging_dir,
    remove_staging_dir,
)

_AUDIT_PATH = Path(__file__).resolve().parents[1] / "config/multistate_response_behavior_adversarial_audit_v1.json"


def publish_multistate_behavior_shadow(
    preview: VerifiedMultistateBehaviorShadow,
    *,
    out_dir: Path,
    overwrite: bool,
) -> dict[str, object]:
    """Materialize one digest-bound shadow bundle and publish it atomically."""

    final_dir = Path(out_dir).resolve()
    stage = create_staging_dir(final_dir, overwrite=overwrite)
    try:
        tables_dir = stage / "tables"
        tables_dir.mkdir()
        normalization_path = stage / "normalization.json"
        normalization_path.write_text(
            json.dumps(preview.normalization_record, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        frames = _publication_frames(preview)
        artifacts: dict[str, Path] = {"normalization": normalization_path}
        for table_id, frame in frames.items():
            path = tables_dir / f"{table_id}.parquet"
            frame.to_parquet(path, index=False)
            artifacts[f"table__{table_id}"] = path
        plot_artifacts = render_multistate_behavior_plots(
            normalization_sensitivity=preview.completion.normalization_sensitivity,
            grouped_validation=preview.completion.grouped_objective_validation,
            allocation_comparison=preview.completion.allocation_comparison,
            prediction_scores=preview.evidence.prediction_scores,
            output_dir=stage / "plots",
        )
        artifacts.update(plot_artifacts)
        audit = load_strict_behavior_json(_AUDIT_PATH)
        verify_behavior_adversarial_audit_record(audit)
        audit_path = stage / "independent_adversarial_audit.json"
        audit_path.write_text(
            json.dumps(audit, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        artifacts["independent_adversarial_audit"] = audit_path
        source_equivalence_path = stage / "source_equivalence.json"
        source_equivalence_path.write_text(
            json.dumps(
                build_source_equivalence_receipt(preview),
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        artifacts["source_equivalence"] = source_equivalence_path
        preliminary_inventory = artifact_inventory(stage, artifacts)
        decision = build_multistate_behavior_decision(
            preview,
            artifact_inventory=preliminary_inventory,
            independent_audit=audit,
            independent_audit_sha256=str(preliminary_inventory["independent_adversarial_audit"]["sha256"]),
        )
        decision_path = stage / "decision.json"
        decision_path.write_text(
            json.dumps(decision, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        artifacts["decision"] = decision_path
        report_path = stage / "report.md"
        report_path.write_text(
            render_multistate_behavior_report(
                decision,
                grouped_validation=preview.completion.grouped_objective_validation,
                allocation_comparison=preview.completion.allocation_comparison,
                hard_behavior_summary=preview.hard_comparison.summary,
                observed_control_face_validity=preview.completion.observed_control_face_validity,
                family_cardinality_pressure=preview.completion.family_cardinality_pressure,
            ),
            encoding="utf-8",
        )
        artifacts["report"] = report_path
        inventory = artifact_inventory(stage, artifacts)
        manifest = _manifest(preview, inventory=inventory, frames=frames)
        manifest_path = stage / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        verify_multistate_behavior_shadow(stage)
        publish_staging_dir(stage, final_dir, overwrite=overwrite)
    except Exception:
        remove_staging_dir(stage)
        raise
    return verify_multistate_behavior_shadow(final_dir)


def _publication_frames(preview: VerifiedMultistateBehaviorShadow) -> dict[str, pd.DataFrame]:
    evidence = preview.evidence
    return {
        "normalization_response_resolution": preview.normalization.response_resolution_rows,
        "normalization_signal_resolution": preview.normalization.signal_resolution_rows,
        "observed_scores": evidence.observed_scores,
        "observed_coordinates": evidence.observed_coordinates,
        "bootstrap_scores": evidence.bootstrap_scores,
        "bootstrap_rank_stability": evidence.bootstrap_rank_stability,
        "bootstrap_rank_draws": evidence.bootstrap_rank_draws,
        "event_sensitivity": evidence.event_sensitivity,
        "repeated_candidate_agreement": evidence.repeated_candidate_agreement,
        "censor_exclusions": preview.censor_exclusions,
        "prediction_scores": evidence.prediction_scores,
        "prediction_surface_diagnostics": build_prediction_surface_diagnostics(evidence.prediction_scores),
        "hard_behavior_summary": preview.hard_comparison.summary,
        "hard_behavior_detail": preview.hard_comparison.detail,
        "normalization_sensitivity": preview.completion.normalization_sensitivity,
        "grouped_objective_validation": preview.completion.grouped_objective_validation,
        "allocation_comparison": preview.completion.allocation_comparison,
        "observed_control_face_validity": preview.completion.observed_control_face_validity,
        "family_cardinality_pressure": preview.completion.family_cardinality_pressure,
        "grouped_rmf_resolution": preview.completion.rmf_resolution_rows,
        "rmf_replay_calibration": preview.completion.rmf_replay_calibration,
        "prediction_vectors": preview.completion.prediction_vectors,
    }


def _manifest(
    preview: VerifiedMultistateBehaviorShadow,
    *,
    inventory: dict[str, dict[str, object]],
    frames: dict[str, pd.DataFrame],
) -> dict[str, object]:
    receipt = preview.normalization.verified_cohort_receipt
    if receipt is None:
        raise ValueError("shadow publication requires an exhaustive cohort receipt.")
    return {
        "schema_id": SCHEMA_ID,
        "schema_version": "1",
        "study_id": preview.normalization.protocol.study_id,
        "protocol_id": preview.normalization.protocol.protocol_id,
        "status": "shadow_only",
        "activation": {"campaign": "prohibited", "synthesis": "prohibited"},
        "objective_name": preview.normalization.protocol.objective_name,
        "comparator": {
            "objective_name": preview.normalization.protocol.comparator_objective_name,
            "score_channel": preview.normalization.protocol.comparator_score_channel,
            "direction": preview.normalization.protocol.comparator_direction,
            "comparison_role": preview.normalization.protocol.comparison_role,
        },
        "source": preview.source,
        "normalization_source": preview.normalization_record["source"],
        "cohort": {
            "cohort_id": receipt.cohort_id,
            "unit_count": receipt.unit_count,
            "candidate_count": receipt.candidate_count,
            "reader_experiment_count": receipt.reader_experiment_count,
            "unit_ids_sha256": receipt.unit_ids_sha256,
            "source_rows_sha256": receipt.source_rows_sha256,
        },
        "excluded_nonexact_unit_count": receipt.excluded_nonexact_unit_count,
        "decision": {
            "promotion_decision": "no_go",
            "campaign_activation": "prohibited",
            "synthesis": "prohibited",
        },
        "tables": {
            table_id: {"rows": len(frame), "columns": list(frame.columns)} for table_id, frame in frames.items()
        },
        "artifacts": inventory,
        "claim_boundary": "shadow_evidence_only_no_campaign_activation_or_synthesis_authorization",
    }


__all__ = ["SCHEMA_ID", "publish_multistate_behavior_shadow", "verify_multistate_behavior_shadow"]
