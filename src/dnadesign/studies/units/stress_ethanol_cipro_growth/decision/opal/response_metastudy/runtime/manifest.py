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

import numpy as np
import pandas as pd

from dnadesign.opal import RESPONSE_MAGNITUDE_FEASIBILITY_API_VERSION

from ..core.contracts import (
    SFXI_SOURCE_PROVENANCE,
    STRESS_STATE_IDS,
    MetastudyPaths,
    PolicySpec,
    SfxiEvidenceFrame,
    StressCampaignContract,
)
from .publication import source_inventory
from .reader_response_bundle import ReaderResponseBundle


def write_metastudy_manifest(
    *,
    paths: MetastudyPaths,
    sfxi_evidence: tuple[SfxiEvidenceFrame, ...],
    stress_campaign: StressCampaignContract,
    reader_bundle: ReaderResponseBundle,
    policy_specs: Sequence[PolicySpec],
    top_k: int,
    training_matrix_sha256: str,
    recommendation: dict[str, object],
    canonical_sfxi_validation: dict[str, object],
    artifact_records: dict[str, object],
    predictor_parity: dict[str, object],
    grouped_model_validation_summary: dict[str, object],
    shuffled_model_validation_summary: dict[str, object],
    sfxi_comparison: pd.DataFrame,
    response_metric_screen: dict[str, object],
) -> dict[str, object]:
    manifest = {
        "schema_version": "stress_ethanol_cipro_growth.response_metastudy.v7",
        "canonical_sfxi_sources": {
            "documentation": "src/dnadesign/opal/docs/plugins/objectives/sfxi.md",
            "math_helpers": "src/dnadesign/opal/src/objectives/sfxi_math.py",
            "objective_plugin": "src/dnadesign/opal/src/objectives/sfxi_v1.py",
        },
        "candidate_objective": {
            "name": "response_magnitude_feasibility_v1",
            "public_api_version": RESPONSE_MAGNITUDE_FEASIBILITY_API_VERSION,
            "status": "implemented_inactive_pending_label_aggregation_and_opal_promotion",
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
                "status": "configured_inactive_pending_label_promotion",
            },
            "target_views": [
                {
                    "selection_view_id": target_view.id,
                    "label": target_view.label,
                    "target_mask": list(target_view.target_mask),
                }
                for target_view in stress_campaign.target_views
            ],
            "candidate_identity_binding": {
                "runtime_posture": "sfxi_source_provenance_only",
                "promotion_schema_id": "dnadesign.study.promoter_candidate_bindings.v1",
                "promotion_schema_version": "1",
                "study_id": "stress_ethanol_cipro_growth",
                "record_id": "promoter_candidate_bindings/bindings",
                "reader_alias_namespace": "reader.design_id",
                "consumed_for_promotion": False,
            },
            "state_order": list(STRESS_STATE_IDS),
            "training_matrix_sha256": training_matrix_sha256,
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
        "sfxi_comparison": _sfxi_comparison_manifest(sfxi_comparison),
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
    files.append(package_root / "config/reader_response_window.yaml")
    files.extend(package_root.rglob("*.py"))
    files.append(stress_campaign.config_path)
    for source in SFXI_SOURCE_PROVENANCE:
        campaign_dir = paths.campaign_root / source.source_campaign_slug
        ledger_dir = campaign_dir / "outputs/ledger"
        files.append(campaign_dir / "inputs/r0/reader_vec8_batch0.csv")
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


def _sfxi_comparison_manifest(comparison: pd.DataFrame) -> dict[str, object]:
    alternatives = comparison.loc[~comparison["assay_summary_id"].eq("snapshot_12h")]
    finite_score = alternatives.loc[
        np.isfinite(alternatives["score_spearman_to_snapshot"]),
        "score_spearman_to_snapshot",
    ]
    support_ranges = {
        str(selection_view_id): {
            "min": int(frame["logic_support_count"].min()),
            "max": int(frame["logic_support_count"].max()),
        }
        for selection_view_id, frame in comparison.groupby("selection_view_id", sort=True)
    }
    return {
        "baseline": "snapshot_12h",
        "time_basis": "reader_declared_event_relative_h",
        "alternative_summary_count": int(alternatives["assay_summary_id"].nunique()),
        "window_specs": sorted(alternatives["assay_summary_id"].astype(str).unique().tolist()),
        "minimum_finite_score_spearman_to_snapshot": (float(finite_score.min()) if not finite_score.empty else None),
        "logic_support_range_by_selection_view": support_ranges,
        "recommended_secondary_summary_id": None,
        "next_candidate_time_basis": None,
        "next_candidate_summary": None,
        "interpretation": (
            "Reader event-relative reductions are compared with the immutable snapshot only as canonical "
            "SFXI reporting overlays; the response metric consumes the unscaled state summaries."
        ),
        "verdict": "screen_only_no_label_promotion",
        "guardrail": (
            "Reader event-relative summaries are verified assay records, but they are not OPAL labels or a new "
            "SFXI objective until the study explicitly promotes one candidate-level label contract."
        ),
    }
