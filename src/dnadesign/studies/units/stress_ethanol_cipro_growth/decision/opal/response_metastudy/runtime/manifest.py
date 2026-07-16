"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/runtime/manifest.py

Provenance-backed manifest construction for the response metric metastudy.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from dnadesign.opal import RESPONSE_MAGNITUDE_FEASIBILITY_API_VERSION
from dnadesign.studies.units.stress_ethanol_cipro_growth.promoter_candidate_bindings import (
    BINDINGS_RECORD_ID,
    READER_ALIAS_NAMESPACE,
    SCHEMA_ID,
    SCHEMA_VERSION,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations.reader_bundle import (
    ReaderResponseBundle,
)

from ...source_evidence import sfxi_round0_source_evidence_dir
from ..core.contracts import (
    SFXI_SOURCE_PROVENANCE,
    STRESS_STATE_IDS,
    MetastudyPaths,
    PolicySpec,
    SfxiEvidenceFrame,
    StressCampaignContract,
)
from .candidate_identity import ResponseCandidateIdentityBindings
from .label_truth import LABEL_TRUTH_SOURCE, LabelTruthState
from .measurement_selection import (
    SCHEMA_ID as RESPONSE_SELECTION_SCHEMA_ID,
)
from .measurement_selection import (
    SCHEMA_VERSION as RESPONSE_SELECTION_SCHEMA_VERSION,
)
from .measurement_selection import (
    ResponseMeasurementSelection,
)
from .publication import METASTUDY_SCHEMA_VERSION, source_inventory


def build_label_truth_record(
    state: LabelTruthState,
    *,
    screen_source_scope: str,
    screen_source_label_truth_role: str,
) -> dict[str, object]:
    """Project verified label readiness without granting screen rows authority."""

    promotion = state.observed_label_promotion_manifest
    return {
        "state": state.state,
        "source": LABEL_TRUTH_SOURCE,
        "screen_source_scope": screen_source_scope,
        "screen_source_label_truth_role": screen_source_label_truth_role,
        "label_source_state": state.label_source_state,
        "observed_label_promotion_manifest": None if promotion is None else dict(promotion),
    }


def write_metastudy_manifest(
    *,
    paths: MetastudyPaths,
    sfxi_evidence: tuple[SfxiEvidenceFrame, ...],
    stress_campaign: StressCampaignContract,
    reader_bundle: ReaderResponseBundle,
    measurement_selection: ResponseMeasurementSelection,
    label_truth_state: LabelTruthState,
    candidate_identity_bindings: ResponseCandidateIdentityBindings,
    policy_specs: Sequence[PolicySpec],
    top_k: int,
    sfxi_training_matrix_sha256: str,
    response_x_matrix_sha256: str,
    recommendation: dict[str, object],
    canonical_sfxi_validation: dict[str, object],
    artifact_records: dict[str, object],
    predictor_parity: dict[str, object],
    grouped_model_validation_summary: dict[str, object],
    shuffled_model_validation_summary: dict[str, object],
    response_metric_screen: dict[str, object],
) -> dict[str, object]:
    label_truth = build_label_truth_record(
        label_truth_state,
        screen_source_scope=measurement_selection.scope,
        screen_source_label_truth_role=measurement_selection.label_truth_role,
    )
    decision_gates = {
        "label_truth_ready": label_truth_state.ready,
        "model_support_ready": bool(response_metric_screen["model_support_ready"]),
        "selection_policy_promoted": False,
        "synthesis_authorized": False,
        "posture": "retrospective_screen_only",
        "opal_operational_state_included": False,
    }
    manifest = {
        "schema_version": METASTUDY_SCHEMA_VERSION,
        "label_truth": label_truth,
        "decision_gates": decision_gates,
        "canonical_sfxi_sources": {
            "documentation": "src/dnadesign/opal/docs/plugins/objectives/sfxi.md",
            "math_helpers": "src/dnadesign/opal/src/objectives/sfxi_math.py",
            "objective_plugin": "src/dnadesign/opal/src/objectives/sfxi_v1.py",
        },
        "candidate_objective": {
            "name": "response_magnitude_feasibility_v1",
            "public_api_version": RESPONSE_MAGNITUDE_FEASIBILITY_API_VERSION,
            "status": "implemented",
            "documentation": "src/dnadesign/opal/docs/plugins/objectives/response-magnitude-feasibility.md",
            "public_api": "dnadesign.opal",
            "study_decision": (
                "docs/studies/stress_ethanol_cipro_growth/contexts/opal/response-magnitude-feasibility.md"
            ),
        },
        "source": {
            "sfxi_source_provenance": [
                {
                    "source_id": evidence.source.source_id,
                    "source_campaign_slug": evidence.source.source_campaign_slug,
                    "lifecycle": evidence.source.lifecycle,
                    "selection_view_id": evidence.target_view.id,
                    "run_id": evidence.run_id,
                    "rows": int(len(evidence.predictions)),
                    "denom": float(evidence.denom),
                }
                for evidence in sfxi_evidence
            ],
            "stress_campaign": {
                "slug": stress_campaign.slug,
                "config": source_inventory(paths.repo_root, [stress_campaign.config_path])[0],
            },
            "target_views": [
                {
                    "selection_view_id": target_view.id,
                    "label": target_view.label,
                    "target_mask": list(target_view.target_mask),
                    "response_semantics": target_view.response_semantics,
                }
                for target_view in stress_campaign.target_views
            ],
            "candidate_identity_binding": {
                "role": "candidate_identity_authority",
                "schema_id": SCHEMA_ID,
                "schema_version": SCHEMA_VERSION,
                "study_id": "stress_ethanol_cipro_growth",
                "record_id": BINDINGS_RECORD_ID,
                "reader_alias_namespace": READER_ALIAS_NAMESPACE,
                "binding_count": candidate_identity_bindings.binding_count,
                "candidate_count": candidate_identity_bindings.candidate_count,
                "resolved_model_screen_candidate_count": len(candidate_identity_bindings.rows),
                "declared_unbound_reader_design_count": candidate_identity_bindings.excluded_design_count,
                "files": source_inventory(
                    candidate_identity_bindings.bundle_root,
                    [candidate_identity_bindings.manifest_path, candidate_identity_bindings.records_path],
                ),
            },
            "response_measurement_selection": {
                "schema_id": RESPONSE_SELECTION_SCHEMA_ID,
                "schema_version": RESPONSE_SELECTION_SCHEMA_VERSION,
                "selection_id": measurement_selection.selection_id,
                "scope": measurement_selection.scope,
                "label_truth_role": measurement_selection.label_truth_role,
                "row_count": len(measurement_selection.rows),
                "excluded_designs": measurement_selection.excluded_designs.to_dict(orient="records"),
                "config": source_inventory(paths.repo_root, [measurement_selection.config_path])[0],
            },
            "state_order": list(STRESS_STATE_IDS),
            "sfxi_training_matrix_sha256": sfxi_training_matrix_sha256,
            "response_x_matrix_sha256": response_x_matrix_sha256,
            "files": source_inventory(paths.repo_root, _provenance_files(paths, stress_campaign=stress_campaign)),
            "reader_bundle": _reader_bundle_inventory(reader_bundle),
        },
        "policy_count": int(len(policy_specs)),
        "top_k": int(top_k),
        "thresholds": recommendation["thresholds"],
        "canonical_sfxi_recompute": canonical_sfxi_validation,
        "recommendation": recommendation,
        "artifacts": artifact_records,
        "guardrail": "Do not treat predicted SFXI scores as biological validation.",
        "predictor_parity": predictor_parity,
        "model_validation": {
            "promotion_gate": grouped_model_validation_summary,
            "descriptive_shuffled": shuffled_model_validation_summary,
        },
        "response_metric_screen": response_metric_screen,
    }
    manifest_path = paths.out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if manifest_path.stat().st_size <= 0:
        raise RuntimeError("Generated manifest is empty.")
    return manifest


def _provenance_files(
    paths: MetastudyPaths,
    *,
    stress_campaign: StressCampaignContract,
) -> tuple[Path, ...]:
    files: list[Path] = [
        paths.repo_root / "src/dnadesign/opal/docs/plugins/objectives/sfxi.md",
        paths.repo_root / "src/dnadesign/opal/src/objectives/sfxi_math.py",
        paths.repo_root / "src/dnadesign/opal/src/objectives/sfxi_v1.py",
        paths.repo_root / "src/dnadesign/opal/docs/plugins/objectives/response-magnitude-feasibility.md",
        paths.repo_root / "src/dnadesign/opal/src/objectives/response_magnitude_feasibility_v1.py",
        paths.repo_root / "src/dnadesign/opal/api/sfxi.py",
        paths.repo_root / "docs/studies/stress_ethanol_cipro_growth/contexts/opal/response-metastudy.md",
        paths.repo_root / "docs/studies/stress_ethanol_cipro_growth/contexts/opal/response-magnitude-feasibility.md",
        paths.repo_root / "docs/studies/stress_ethanol_cipro_growth/contexts/opal/sfxi-round0-source-evidence.md",
    ]
    package_root = Path(__file__).resolve().parents[1]
    files.append(package_root / "README.md")
    files.append(
        paths.repo_root
        / "src/dnadesign/studies/units/stress_ethanol_cipro_growth"
        / "response_window_observations/config/reader_response_window.yaml"
    )
    files.extend(package_root.rglob("*.py"))
    files.append(stress_campaign.config_path)
    for source in SFXI_SOURCE_PROVENANCE:
        source_dir = sfxi_round0_source_evidence_dir(
            paths.repo_root,
            source_slug=source.source_campaign_slug,
        )
        ledger_dir = source_dir / "outputs/ledger"
        files.append(source_dir / "inputs/r0/reader_vec8_batch0.csv")
        for dataset_name in ("runs.parquet", "labels.parquet", "predictions"):
            files.extend((ledger_dir / dataset_name).rglob("*.parquet"))
    return tuple(files)


def _reader_bundle_inventory(bundle: ReaderResponseBundle) -> dict[str, object]:
    return {
        "root": str(bundle.root),
        "manifest": source_inventory(bundle.root, [bundle.manifest_path])[0],
        "schema_version": bundle.manifest["schema_version"],
        "request_id": bundle.manifest["request_id"],
        "request": bundle.manifest["request"],
        "primary_reduction_id": bundle.primary_reduction_id,
        "contracts": bundle.manifest["contracts"],
        "counts": bundle.manifest["counts"],
        "source_records": bundle.manifest["source_records"],
    }
