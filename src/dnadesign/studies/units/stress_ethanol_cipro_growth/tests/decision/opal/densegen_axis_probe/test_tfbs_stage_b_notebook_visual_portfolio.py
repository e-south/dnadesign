from __future__ import annotations

import json
from pathlib import Path

import pytest

from .probe_modules import probe_module

_portfolio = probe_module("tfbs.stage_b.notebook_visuals.portfolio")
TfbsStageBReviewPortfolioSource = _portfolio.TfbsStageBReviewPortfolioSource
write_tfbs_stage_b_review_portfolio = _portfolio.write_tfbs_stage_b_review_portfolio


def test_review_portfolio_namespaces_overlapping_replicated_sets(tmp_path: Path) -> None:
    count_preserving = _write_replicated_review_source(
        tmp_path,
        source_id="slot_position_count_preserving",
        label_name="lexA_in_slot0",
    )
    count_fixed = _write_replicated_review_source(
        tmp_path,
        source_id="slot_position_count_fixed",
        label_name="lexA_in_slot0",
        control_role="count_fixed_shuffled_slot_negative_control",
    )

    result = write_tfbs_stage_b_review_portfolio(
        [
            TfbsStageBReviewPortfolioSource(
                surface_id="slot_position_count_preserving",
                surface_label="Count-preserving slot diagnostic",
                evidence_tier="diagnostic",
                review_summary_json_path=count_preserving,
            ),
            TfbsStageBReviewPortfolioSource(
                surface_id="slot_position_count_fixed",
                surface_label="Count-fixed slot-position sentinel",
                evidence_tier="current_boundary",
                review_summary_json_path=count_fixed,
            ),
        ],
        out_dir=tmp_path / "portfolio",
        collection_id="tfbs_probe_portfolio",
    )

    index = json.loads(result.collection_visual_index_path.read_text(encoding="utf-8"))
    assert index["schema_version"] == "opal.collection_visual_manifest_index.v1"
    assert index["comparison_set_count"] == 2
    assert index["visual_count"] == 2
    assert index["surface_kinds"] == ["study_realized_label_review"]
    assert [item["key"] for item in index["comparison_sets"]] == [
        "slot_position_count_preserving__stage_b_realized_label_review__lexA_in_slot0",
        "slot_position_count_fixed__stage_b_realized_label_review__lexA_in_slot0",
    ]
    assert [item["label"] for item in index["comparison_sets"]] == [
        "Count-preserving slot diagnostic: LexA in slot 0 DenseGen label vs slot-position control pair",
        "Count-fixed slot-position sentinel: LexA in slot 0 DenseGen label vs count-fixed shuffled-slot control pair",
    ]
    assert [item["evidence_tier"] for item in index["comparison_sets"]] == ["diagnostic", "current_boundary"]
    assert index["evidence_tiers"] == [
        {"id": "current_boundary", "label": "Current boundary", "rank": 20},
        {"id": "diagnostic", "label": "Diagnostic", "rank": 70},
    ]
    assert {visual["source_review_surface_id"] for visual in index["visuals"]} == {
        "slot_position_count_preserving",
        "slot_position_count_fixed",
    }
    assert {
        (visual["source_review_surface_id"], visual["evidence_tier"], visual["evidence_tier_label"])
        for visual in index["visuals"]
    } == {
        ("slot_position_count_preserving", "diagnostic", "Diagnostic"),
        ("slot_position_count_fixed", "current_boundary", "Current boundary"),
    }
    diagnostic_visual = next(
        visual for visual in index["visuals"] if visual["source_review_surface_id"] == "slot_position_count_preserving"
    )
    assert diagnostic_visual["premise"].startswith("Diagnostic surface:")
    assert "Active selection should enrich" not in diagnostic_visual["premise"]
    assert diagnostic_visual["claim_boundary"].startswith("Diagnostic only:")
    assert diagnostic_visual["interpretation_note"].startswith("This diagnostic surface remains visible")
    assert all(Path(visual["path"]).exists() for visual in index["visuals"])

    collection = json.loads(result.collection_manifest_path.read_text(encoding="utf-8"))
    assert collection == {
        "schema_version": "opal.campaign_collection.v2",
        "collection_id": "tfbs_probe_portfolio",
        "dimensions": [{"id": "target", "label": "TFBS label"}],
        "relationships": [],
        "comparison_views": [],
        "collection_visual_surface_kinds": ["study_realized_label_review"],
        "evidence_tiers": [
            {"id": "current_boundary", "label": "Current boundary", "rank": 20},
            {"id": "diagnostic", "label": "Diagnostic", "rank": 70},
        ],
    }


def test_review_portfolio_rejects_nonreplicated_review_sources(tmp_path: Path) -> None:
    summary_path = _write_replicated_review_source(
        tmp_path,
        source_id="single_seed",
        label_name="lexA_in_slot0",
        replicate_count=1,
        replicate_seeds=[7],
    )

    with pytest.raises(ValueError, match="replicated"):
        write_tfbs_stage_b_review_portfolio(
            [
                TfbsStageBReviewPortfolioSource(
                    surface_id="single_seed",
                    surface_label="Single seed",
                    evidence_tier="diagnostic",
                    review_summary_json_path=summary_path,
                )
            ],
            out_dir=tmp_path / "portfolio",
            collection_id="tfbs_probe_portfolio",
        )


def test_review_portfolio_rejects_unknown_evidence_tier(tmp_path: Path) -> None:
    summary_path = _write_replicated_review_source(
        tmp_path,
        source_id="bad_tier",
        label_name="lexA_in_slot0",
    )

    with pytest.raises(ValueError, match="evidence_tier"):
        write_tfbs_stage_b_review_portfolio(
            [
                TfbsStageBReviewPortfolioSource(
                    surface_id="bad_tier",
                    surface_label="Bad tier",
                    evidence_tier="currentish",
                    review_summary_json_path=summary_path,
                )
            ],
            out_dir=tmp_path / "portfolio",
            collection_id="tfbs_probe_portfolio",
        )


def test_review_portfolio_rejects_limited_source_in_current_boundary_tier(tmp_path: Path) -> None:
    summary_path = _write_replicated_review_source(
        tmp_path,
        source_id="limited_boundary",
        label_name="lexA_in_slot0",
        profile_role="boundary_stage_b_sentinel_probe",
        claim_ready=False,
    )

    with pytest.raises(ValueError, match="evidence tier/profile_role mismatch"):
        write_tfbs_stage_b_review_portfolio(
            [
                TfbsStageBReviewPortfolioSource(
                    surface_id="limited_boundary",
                    surface_label="Limited boundary",
                    evidence_tier="current_boundary",
                    review_summary_json_path=summary_path,
                )
            ],
            out_dir=tmp_path / "portfolio",
            collection_id="tfbs_probe_portfolio",
        )


def test_review_portfolio_rejects_unready_current_claim_source(tmp_path: Path) -> None:
    summary_path = _write_replicated_review_source(
        tmp_path,
        source_id="unready_claim",
        label_name="lexA_count_fraction",
        profile_role="canonical_stage_b_probe",
        claim_ready=False,
    )

    with pytest.raises(ValueError, match="not claim-ready"):
        write_tfbs_stage_b_review_portfolio(
            [
                TfbsStageBReviewPortfolioSource(
                    surface_id="unready_claim",
                    surface_label="Unready claim",
                    evidence_tier="current_claim",
                    review_summary_json_path=summary_path,
                )
            ],
            out_dir=tmp_path / "portfolio",
            collection_id="tfbs_probe_portfolio",
        )


def _write_replicated_review_source(
    tmp_path: Path,
    *,
    source_id: str,
    label_name: str,
    replicate_count: int = 3,
    replicate_seeds: list[int] | None = None,
    control_role: str | None = None,
    profile_role: str | None = None,
    claim_ready: bool | None = None,
) -> Path:
    root = tmp_path / source_id
    root.mkdir(parents=True)
    trajectory_path = root / "trajectory.csv"
    pair_summary_path = root / "pair_summary.csv"
    trajectory_path.write_text("label_name,round,value\nlexA_in_slot0,0,1\n", encoding="utf-8")
    pair_summary_path.write_text("label_name,value\nlexA_in_slot0,1\n", encoding="utf-8")
    plot_path = root / f"{label_name}.png"
    plot_path.write_bytes(b"not-a-real-png-but-existing")
    plot_manifest_path = root / "plot_manifest.json"
    plot_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "stress_ethanol_cipro_growth.tfbs_stage_b_review_plots.v1",
                "plot_count": 1,
                "plots": [
                    {
                        "kind": "realized_label_lift_trajectory",
                        "label_name": label_name,
                        "path": str(plot_path),
                        "interval_kind": "sample_sd",
                        **({"control_role": control_role} if control_role is not None else {}),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    summary_path = root / "summary.json"
    resolved_profile_role = profile_role or (
        "boundary_stage_b_count_fixed_sentinel_probe"
        if control_role == "count_fixed_shuffled_slot_negative_control"
        else "boundary_stage_b_sentinel_probe"
    )
    resolved_claim_ready = (
        claim_ready if claim_ready is not None else control_role == "count_fixed_shuffled_slot_negative_control"
    )
    claim_readiness = (
        {
            "blocked_or_limited_claim_count": 0,
            "blocked_or_limited_labels": [],
            "claim_readiness_status_counts": {"READY_AS_REPLICATED_VALID_NULL_LEARNABILITY_SIGNAL": 1},
            "ready_claim_count": 1,
            "ready_labels": [label_name],
        }
        if resolved_claim_ready
        else {
            "blocked_or_limited_claim_count": 1,
            "blocked_or_limited_labels": [label_name],
            "claim_readiness_status_counts": {"LIMITED_INVALID_NEGATIVE_CONTROL_REPLICATE": 1},
            "ready_claim_count": 0,
            "ready_labels": [],
        }
    )
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": "stress_ethanol_cipro_growth.tfbs_stage_b_replicated_review.v1",
                "status": "PASS",
                "claim_readiness": claim_readiness,
                "interpretation_boundary": "Fixture interpretation boundary.",
                "replicate_count": replicate_count,
                "replicate_seeds": replicate_seeds or [7, 17, 29],
                "target_profile": {
                    "profile_id": "fixture_profile",
                    "profile_role": resolved_profile_role,
                    "label_names": [label_name],
                    "label_family_ids": ["tf_slot_family_presence"],
                    "canonical": False,
                    "interpretation_boundary": "Fixture interpretation boundary.",
                },
                "trajectory_csv_path": str(trajectory_path),
                "replicate_pair_summary_csv_path": str(pair_summary_path),
                "plot_manifest_json_path": str(plot_manifest_path),
            }
        ),
        encoding="utf-8",
    )
    return summary_path
