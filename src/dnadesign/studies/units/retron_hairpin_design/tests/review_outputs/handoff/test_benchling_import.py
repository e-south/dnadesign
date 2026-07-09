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

from dnadesign.studies.units.retron_hairpin_design.compiler.exceptions import RetronMsdCompilerError
from dnadesign.studies.units.retron_hairpin_design.review_outputs.service import (
    generate_retron_hairpin_review_outputs,
)

from ...support.paths import repo_root_from
from ...support.review_outputs import fake_video_writer, write_fake_materialized_bundle
from ...support.review_plans import read_review_plan, write_review_plan, write_review_plan_with_test_pwm


def test_benchling_import_uses_deliverable_plan_assigned_ids(tmp_path: Path) -> None:
    repo_root = repo_root_from(__file__)
    plan_path = write_review_plan_with_test_pwm(tmp_path / "plan", repo_root=repo_root)
    plan = read_review_plan(plan_path)
    benchling = plan["artifact_families"]["benchling_genbank_import"]
    review_ids = plan["artifact_families"]["msd_sequence_review_stills"]["review_variant_ids"]
    benchling["assigned_retron_ids"]["r26-w02-17"] = "pES-retron-205"
    benchling["record_ids"]["r26-w02-17"] = "msd-retron-205"
    benchling["expected_files"][0] = "benchling_genbank/msd-retron-205.gb"
    review_ids["r26-w02-17"] = "pES-retron-205"
    write_review_plan(plan_path, plan)

    materialized_root = write_fake_materialized_bundle(tmp_path / "materialized", repo_root=repo_root)
    result = generate_retron_hairpin_review_outputs(
        deliverable_plan_path=plan_path,
        materialized_root=materialized_root,
        out_dir=tmp_path / "outputs",
        repo_root=repo_root,
        video_writer=fake_video_writer,
    )

    assert (result.benchling_genbank_dir / "msd-retron-205.gb").is_file()
    assert not (result.benchling_genbank_dir / "msd-retron-195.gb").exists()


def test_benchling_import_fails_on_expected_file_drift(tmp_path: Path) -> None:
    repo_root = repo_root_from(__file__)
    plan_path = write_review_plan_with_test_pwm(tmp_path / "plan", repo_root=repo_root)
    plan = read_review_plan(plan_path)
    plan["artifact_families"]["benchling_genbank_import"]["expected_files"][0] = "benchling_genbank/msd-retron-999.gb"
    write_review_plan(plan_path, plan)
    materialized_root = write_fake_materialized_bundle(tmp_path / "materialized", repo_root=repo_root)

    with pytest.raises(RetronMsdCompilerError, match="expected_files must match record_ids"):
        generate_retron_hairpin_review_outputs(
            deliverable_plan_path=plan_path,
            materialized_root=materialized_root,
            out_dir=tmp_path / "outputs",
            repo_root=repo_root,
            video_writer=fake_video_writer,
        )


def test_benchling_import_fails_on_record_id_drift(tmp_path: Path) -> None:
    repo_root = repo_root_from(__file__)
    plan_path = write_review_plan_with_test_pwm(tmp_path / "plan", repo_root=repo_root)
    plan = read_review_plan(plan_path)
    record_ids = plan["artifact_families"]["benchling_genbank_import"]["record_ids"]
    record_ids["r99-w02-17"] = record_ids.pop("r26-w02-17")
    write_review_plan(plan_path, plan)
    materialized_root = write_fake_materialized_bundle(tmp_path / "materialized", repo_root=repo_root)

    with pytest.raises(RetronMsdCompilerError, match="record_ids must match assigned_retron_ids"):
        generate_retron_hairpin_review_outputs(
            deliverable_plan_path=plan_path,
            materialized_root=materialized_root,
            out_dir=tmp_path / "outputs",
            repo_root=repo_root,
            video_writer=fake_video_writer,
        )


def test_benchling_import_fails_on_source_precedent_drift(tmp_path: Path) -> None:
    repo_root = repo_root_from(__file__)
    plan_path = write_review_plan_with_test_pwm(tmp_path / "plan", repo_root=repo_root)
    plan = read_review_plan(plan_path)
    precedent_ids = plan["artifact_families"]["benchling_genbank_import"]["source_precedent_ids"]
    precedent_ids["r99-w02-17"] = precedent_ids.pop("r26-w02-17")
    write_review_plan(plan_path, plan)
    materialized_root = write_fake_materialized_bundle(tmp_path / "materialized", repo_root=repo_root)

    with pytest.raises(RetronMsdCompilerError, match="source_precedent_ids must match assigned_retron_ids"):
        generate_retron_hairpin_review_outputs(
            deliverable_plan_path=plan_path,
            materialized_root=materialized_root,
            out_dir=tmp_path / "outputs",
            repo_root=repo_root,
            video_writer=fake_video_writer,
        )
