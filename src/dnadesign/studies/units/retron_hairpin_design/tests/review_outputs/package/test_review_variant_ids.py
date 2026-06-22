"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/tests/review_outputs/package/test_review_variant_ids.py

Tests for tetO trim review-frame retron-id contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dnadesign.studies.units.retron_hairpin_design.compiler.exceptions import RetronMsdCompilerError
from dnadesign.studies.units.retron_hairpin_design.review_outputs.service import (
    generate_teto_pwm_trim_rescue_review_outputs,
)

from ...support.paths import repo_root_from
from ...support.review_outputs import fake_video_writer, write_fake_materialized_bundle
from ...support.review_plans import write_review_plan_with_test_pwm


def test_teto_pwm_trim_review_outputs_fail_fast_on_review_variant_id_drift(tmp_path: Path) -> None:
    repo_root = repo_root_from(__file__)
    materialized_root = write_fake_materialized_bundle(tmp_path / "materialized", repo_root=repo_root)
    plan_path = write_review_plan_with_test_pwm(tmp_path / "plan", repo_root=repo_root)
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    review_ids = plan["artifact_families"]["msd_sequence_review_stills"]["review_variant_ids"]
    review_ids["r26-w02-17"] = "pES-retron-205"
    plan_path.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")

    with pytest.raises(RetronMsdCompilerError, match="must match Benchling assigned_retron_ids"):
        generate_teto_pwm_trim_rescue_review_outputs(
            deliverable_plan_path=plan_path,
            materialized_root=materialized_root,
            out_dir=tmp_path / "outputs",
            repo_root=repo_root,
            video_writer=fake_video_writer,
        )
