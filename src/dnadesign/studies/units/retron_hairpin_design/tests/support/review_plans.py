"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/tests/support/review_plans.py

Review-plan fixtures for Retron hairpin study tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .pwm_fixtures import write_test_tetr_meme_pwm


def write_review_plan_with_test_pwm(root: Path, *, repo_root: Path) -> Path:
    return write_review_plan_fixture(
        root,
        repo_root=repo_root,
        deliverable_plan_id="teto_retained_span_trim_tetr_pwm_elite_v1",
    )


def write_review_plan_fixture(root: Path, *, repo_root: Path, deliverable_plan_id: str) -> Path:
    source_path = (
        repo_root
        / "docs"
        / "studies"
        / "retron_hairpin_design"
        / "workbench"
        / "deliverables"
        / f"{deliverable_plan_id}.yaml"
    )
    plan = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    meme_pwm_path = write_test_tetr_meme_pwm(root / "fixtures" / "tetR__westmann_tetr_mitomi__tetR.meme")
    plan["source_refs"]["meme_pwm"] = str(meme_pwm_path)
    plan_path = root / "deliverables" / f"{deliverable_plan_id}.yaml"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")
    return plan_path


__all__ = ["write_review_plan_fixture", "write_review_plan_with_test_pwm"]
