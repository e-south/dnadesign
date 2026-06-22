"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/tests/review_outputs/handoff/test_benchling_import.py

Tests for tetO trim Benchling GenBank import contracts.

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


def test_benchling_import_uses_deliverable_plan_assigned_ids(tmp_path: Path) -> None:
    repo_root = repo_root_from(__file__)
    plan_path = write_review_plan_with_test_pwm(tmp_path / "plan", repo_root=repo_root)
    plan = _read_plan(plan_path)
    benchling = plan["artifact_families"]["benchling_genbank_import"]
    review_ids = plan["artifact_families"]["msd_sequence_review_stills"]["review_variant_ids"]
    benchling["assigned_retron_ids"]["r26-w02-17"] = "pES-retron-205"
    benchling["expected_files"][0] = "benchling_genbank/pES-retron-205-msd[TetR]-r26-w02-17.gb"
    review_ids["r26-w02-17"] = "pES-retron-205"
    _write_plan(plan_path, plan)

    materialized_root = write_fake_materialized_bundle(tmp_path / "materialized", repo_root=repo_root)
    result = generate_teto_pwm_trim_rescue_review_outputs(
        deliverable_plan_path=plan_path,
        materialized_root=materialized_root,
        out_dir=tmp_path / "outputs",
        repo_root=repo_root,
        video_writer=fake_video_writer,
    )

    assert (result.benchling_genbank_dir / "pES-retron-205-msd[TetR]-r26-w02-17.gb").is_file()
    assert not (result.benchling_genbank_dir / "pES-retron-195-msd[TetR]-r26-w02-17.gb").exists()


def test_benchling_import_fails_on_expected_file_drift(tmp_path: Path) -> None:
    repo_root = repo_root_from(__file__)
    plan_path = write_review_plan_with_test_pwm(tmp_path / "plan", repo_root=repo_root)
    plan = _read_plan(plan_path)
    plan["artifact_families"]["benchling_genbank_import"]["expected_files"][0] = (
        "benchling_genbank/pES-retron-999-msd[TetR]-r26-w02-17.gb"
    )
    _write_plan(plan_path, plan)
    materialized_root = write_fake_materialized_bundle(tmp_path / "materialized", repo_root=repo_root)

    with pytest.raises(RetronMsdCompilerError, match="expected_files must match assigned_retron_ids"):
        generate_teto_pwm_trim_rescue_review_outputs(
            deliverable_plan_path=plan_path,
            materialized_root=materialized_root,
            out_dir=tmp_path / "outputs",
            repo_root=repo_root,
            video_writer=fake_video_writer,
        )


def test_benchling_import_fails_on_source_precedent_drift(tmp_path: Path) -> None:
    repo_root = repo_root_from(__file__)
    plan_path = write_review_plan_with_test_pwm(tmp_path / "plan", repo_root=repo_root)
    plan = _read_plan(plan_path)
    precedent_ids = plan["artifact_families"]["benchling_genbank_import"]["source_precedent_ids"]
    precedent_ids["r99-w02-17"] = precedent_ids.pop("r26-w02-17")
    _write_plan(plan_path, plan)
    materialized_root = write_fake_materialized_bundle(tmp_path / "materialized", repo_root=repo_root)

    with pytest.raises(RetronMsdCompilerError, match="source_precedent_ids must match assigned_retron_ids"):
        generate_teto_pwm_trim_rescue_review_outputs(
            deliverable_plan_path=plan_path,
            materialized_root=materialized_root,
            out_dir=tmp_path / "outputs",
            repo_root=repo_root,
            video_writer=fake_video_writer,
        )


def _read_plan(path: Path) -> dict[str, object]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _write_plan(path: Path, plan: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")
