"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/densegen_axis_probe/test_tfbs_learning_loop_portfolio_contracts.py

Regression tests for TFBS learning loop portfolio studies units stress ethanol.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from .probe_modules import probe_module
from .stage_b_portfolio_fixtures import write_learning_loop_source, write_replicated_review_source

_portfolio = probe_module("tfbs.stage_b.notebook_visuals.portfolio")
TfbsProbeQuestionLearningLoopSource = _portfolio.TfbsProbeQuestionLearningLoopSource
TfbsProbeQuestionReviewSource = _portfolio.TfbsProbeQuestionReviewSource
write_tfbs_stage_b_review_portfolio = _portfolio.write_tfbs_stage_b_review_portfolio


def test_review_portfolio_rejects_learning_loop_profile_that_does_not_match_tier(tmp_path: Path) -> None:
    count_fraction = write_replicated_review_source(
        tmp_path,
        source_id="count_fraction",
        label_name="lexA_count_fraction",
        profile_role="canonical_stage_b_probe",
        claim_ready=True,
    )
    learning_loop = write_learning_loop_source(
        tmp_path,
        profile_id="tfbs_slot_position_count_fixed_sentinel_probe_v1",
        visual_tier="composition_learning_loop",
    )

    with pytest.raises(ValueError, match="source_profile_ids do not match"):
        write_tfbs_stage_b_review_portfolio(
            [
                TfbsProbeQuestionReviewSource(
                    question_id="count_fraction",
                    question_label="Count-fraction composition",
                    evidence_tier="composition_campaign",
                    review_summary_json_path=count_fraction,
                )
            ],
            learning_loop_sources=[
                TfbsProbeQuestionLearningLoopSource(
                    question_id="count_fraction_learning_loop",
                    question_label="Composition learning loop",
                    evidence_tier="composition_learning_loop",
                    replay_manifest_json_path=learning_loop,
                )
            ],
            out_dir=tmp_path / "portfolio",
            collection_id="tfbs_probe_portfolio",
        )


def test_review_portfolio_rejects_learning_loop_missing_source_profiles(tmp_path: Path) -> None:
    count_fraction = write_replicated_review_source(
        tmp_path,
        source_id="count_fraction",
        label_name="lexA_count_fraction",
        profile_role="canonical_stage_b_probe",
        claim_ready=True,
    )
    learning_loop = write_learning_loop_source(tmp_path)
    payload = json.loads(learning_loop.read_text(encoding="utf-8"))
    payload["source_profile_ids"] = []
    learning_loop.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="source_profile_ids must contain"):
        write_tfbs_stage_b_review_portfolio(
            [
                TfbsProbeQuestionReviewSource(
                    question_id="count_fraction",
                    question_label="Count-fraction composition",
                    evidence_tier="composition_campaign",
                    review_summary_json_path=count_fraction,
                )
            ],
            learning_loop_sources=[
                TfbsProbeQuestionLearningLoopSource(
                    question_id="count_fraction_learning_loop",
                    question_label="Composition learning loop",
                    evidence_tier="composition_learning_loop",
                    replay_manifest_json_path=learning_loop,
                )
            ],
            out_dir=tmp_path / "portfolio",
            collection_id="tfbs_probe_portfolio",
        )
