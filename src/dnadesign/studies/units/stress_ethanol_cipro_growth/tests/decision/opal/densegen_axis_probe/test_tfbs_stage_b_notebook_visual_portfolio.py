from __future__ import annotations

import json
from pathlib import Path

import pytest

from .probe_modules import probe_module
from .stage_b_portfolio_fixtures import write_learning_loop_source, write_replicated_review_source

_portfolio = probe_module("tfbs.stage_b.notebook_visuals.portfolio")
TfbsStageBLearningLoopPortfolioSource = _portfolio.TfbsStageBLearningLoopPortfolioSource
TfbsStageBReviewPortfolioSource = _portfolio.TfbsStageBReviewPortfolioSource
write_tfbs_stage_b_review_portfolio = _portfolio.write_tfbs_stage_b_review_portfolio


def test_review_portfolio_namespaces_overlapping_replicated_sets(tmp_path: Path) -> None:
    count_preserving = write_replicated_review_source(
        tmp_path,
        source_id="slot_position_count_preserving",
        label_name="lexA_in_slot0",
    )
    count_fixed = write_replicated_review_source(
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
        "Count-preserving slot diagnostic: LexA in leftmost slot DenseGen label vs slot-position control pair",
        (
            "Count-fixed slot-position sentinel: LexA in leftmost slot DenseGen label vs "
            "count-fixed shuffled-slot control pair"
        ),
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
    assert "clean negative-control evidence" in diagnostic_visual["claim_boundary"]
    assert diagnostic_visual["interpretation_note"].startswith("This diagnostic remains visible")
    assert "known confounds" in diagnostic_visual["interpretation_note"]
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


def test_review_portfolio_can_include_learning_loop_baseline_source(tmp_path: Path) -> None:
    count_fraction = write_replicated_review_source(
        tmp_path,
        source_id="count_fraction",
        label_name="lexA_count_fraction",
        profile_role="canonical_stage_b_probe",
        claim_ready=True,
    )
    learning_loop = write_learning_loop_source(tmp_path)

    result = write_tfbs_stage_b_review_portfolio(
        [
            TfbsStageBReviewPortfolioSource(
                surface_id="count_fraction",
                surface_label="Count-fraction composition",
                evidence_tier="current_claim",
                review_summary_json_path=count_fraction,
            )
        ],
        learning_loop_sources=[
            TfbsStageBLearningLoopPortfolioSource(
                surface_id="count_fraction_learning_loop",
                surface_label="Count-fraction learning-loop baseline",
                evidence_tier="current_claim",
                replay_manifest_json_path=learning_loop,
            )
        ],
        out_dir=tmp_path / "portfolio",
        collection_id="tfbs_probe_portfolio",
    )

    index = json.loads(result.collection_visual_index_path.read_text(encoding="utf-8"))
    assert index["comparison_set_count"] == 2
    assert index["visual_count"] == 4
    assert sorted(index["surface_kinds"]) == ["study_learning_loop_baseline", "study_realized_label_review"]
    learning_visuals = [
        visual for visual in index["visuals"] if visual["source_review_surface_id"] == "count_fraction_learning_loop"
    ]
    assert len(learning_visuals) == 3
    assert {visual["view_kind"] for visual in learning_visuals} == {
        "frozen_round0_cumulative_enrichment",
        "frozen_round0_endpoint_adaptive_gain",
        "top_budget_signal_recovery",
    }
    assert all(visual["evidence_tier"] == "current_claim" for visual in learning_visuals)
    assert all(Path(visual["path"]).is_absolute() for visual in learning_visuals)
    assert learning_visuals[0]["claim_boundary"].startswith("This supports a harness-level active-learning claim")


def test_review_portfolio_can_include_count_fixed_slot_learning_loop_boundary(tmp_path: Path) -> None:
    count_fixed = write_replicated_review_source(
        tmp_path,
        source_id="slot_position_count_fixed",
        label_name="lexA_in_slot0",
        control_role="count_fixed_shuffled_slot_negative_control",
    )
    learning_loop = write_learning_loop_source(
        tmp_path,
        review_id="count_fixed_slot_learning_loop",
        profile_id="tfbs_slot_position_count_fixed_sentinel_probe_v1",
        visual_tier="current_boundary",
    )

    result = write_tfbs_stage_b_review_portfolio(
        [
            TfbsStageBReviewPortfolioSource(
                surface_id="slot_position_count_fixed",
                surface_label="Count-fixed placement",
                evidence_tier="current_boundary",
                review_summary_json_path=count_fixed,
            )
        ],
        learning_loop_sources=[
            TfbsStageBLearningLoopPortfolioSource(
                surface_id="count_fixed_slot_learning_loop",
                surface_label="Count-fixed placement learning-loop baseline",
                evidence_tier="current_boundary",
                replay_manifest_json_path=learning_loop,
            )
        ],
        out_dir=tmp_path / "portfolio",
        collection_id="tfbs_probe_portfolio",
    )

    index = json.loads(result.collection_visual_index_path.read_text(encoding="utf-8"))
    learning_visuals = [
        visual for visual in index["visuals"] if visual["source_review_surface_id"] == "count_fixed_slot_learning_loop"
    ]
    assert len(learning_visuals) == 3
    assert all(visual["evidence_tier"] == "current_boundary" for visual in learning_visuals)
    assert {visual["comparison_set_match"]["profile_id"] for visual in learning_visuals} == {
        "tfbs_slot_position_count_fixed_sentinel_probe_v1"
    }


def test_review_portfolio_rejects_nonreplicated_review_sources(tmp_path: Path) -> None:
    summary_path = write_replicated_review_source(
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
    summary_path = write_replicated_review_source(
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
    summary_path = write_replicated_review_source(
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
    summary_path = write_replicated_review_source(
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
