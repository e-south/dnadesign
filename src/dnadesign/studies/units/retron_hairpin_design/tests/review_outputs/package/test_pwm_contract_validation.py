"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/tests/review_outputs/package/test_pwm_contract_validation.py

Fail-fast tests for PWM review-output contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dnadesign.studies.units.retron_hairpin_design.compiler.exceptions import RetronMsdCompilerError
from dnadesign.studies.units.retron_hairpin_design.review_outputs.service import (
    generate_retron_hairpin_review_outputs,
)

from ...support.paths import repo_root_from
from ...support.review_outputs import fake_video_writer, write_fake_materialized_bundle
from ...support.review_plans import write_review_plan_with_test_pwm


def test_teto_pwm_trim_review_outputs_fail_fast_on_motif_layer_drift(tmp_path: Path) -> None:
    repo_root = repo_root_from(__file__)
    materialized_root = write_fake_materialized_bundle(tmp_path / "materialized", repo_root=repo_root)
    deliverable_plan_path = write_review_plan_with_test_pwm(tmp_path / "plan", repo_root=repo_root)
    plan = yaml.safe_load(deliverable_plan_path.read_text(encoding="utf-8"))
    plan["artifact_families"]["pwm_trim_review_panel"]["motif_layers"] = [
        plan["artifact_families"]["pwm_trim_review_panel"]["motif_layers"][0]
    ]
    deliverable_plan_path.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        RetronMsdCompilerError,
        match="motif_layers must match design-set parent_payload motif_occurrences",
    ):
        generate_retron_hairpin_review_outputs(
            deliverable_plan_path=deliverable_plan_path,
            materialized_root=materialized_root,
            out_dir=tmp_path / "outputs",
            repo_root=repo_root,
            video_writer=fake_video_writer,
        )
