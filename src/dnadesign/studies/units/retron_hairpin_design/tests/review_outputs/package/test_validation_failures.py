"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/tests/review_outputs/package/test_validation_failures.py

Fail-fast tests for tetO PWM trim rescue review-package inputs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from dnadesign.studies.units.retron_hairpin_design.compiler.exceptions import RetronMsdCompilerError
from dnadesign.studies.units.retron_hairpin_design.review_outputs.service import (
    generate_teto_pwm_trim_rescue_review_outputs,
)

from ...support.paths import repo_root_from
from ...support.review_outputs import fake_video_writer, write_fake_materialized_bundle


def test_teto_pwm_trim_review_outputs_fail_fast_on_wrong_row_count(tmp_path: Path) -> None:
    repo_root = repo_root_from(__file__)
    materialized_root = write_fake_materialized_bundle(tmp_path / "materialized", repo_root=repo_root, row_count=8)

    with pytest.raises(RetronMsdCompilerError, match="Expected 9 materialized sequence rows"):
        _generate(repo_root=repo_root, materialized_root=materialized_root, out_dir=tmp_path / "outputs")


def test_teto_pwm_trim_review_outputs_fail_fast_on_missing_row_artifact(tmp_path: Path) -> None:
    repo_root = repo_root_from(__file__)
    materialized_root = write_fake_materialized_bundle(tmp_path / "materialized", repo_root=repo_root)
    rows = _sequence_index_rows(materialized_root)
    (materialized_root / rows[0]["genbank"]).unlink()

    with pytest.raises(RetronMsdCompilerError, match="Missing materialized review artifact"):
        _generate(repo_root=repo_root, materialized_root=materialized_root, out_dir=tmp_path / "outputs")


def test_teto_pwm_trim_review_outputs_fail_fast_on_non_ok_folding(tmp_path: Path) -> None:
    repo_root = repo_root_from(__file__)
    materialized_root = write_fake_materialized_bundle(tmp_path / "materialized", repo_root=repo_root)
    index_path = materialized_root / "manifest" / "indexes" / "sequence_index.tsv"
    rows = _sequence_index_rows(materialized_root)
    rows[0]["folding_status"] = "backend_missing"
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(RetronMsdCompilerError, match="folding_status == ok"):
        _generate(repo_root=repo_root, materialized_root=materialized_root, out_dir=tmp_path / "outputs")


def test_teto_pwm_trim_review_outputs_fail_fast_on_bad_reverse_complement(tmp_path: Path) -> None:
    repo_root = repo_root_from(__file__)
    materialized_root = write_fake_materialized_bundle(tmp_path / "materialized", repo_root=repo_root)
    rows = _sequence_index_rows(materialized_root)
    (materialized_root / rows[0]["reverse_complement_fasta"]).write_text(
        ">bad_reverse_complement\nAAAAAAAAAA\n",
        encoding="utf-8",
    )

    with pytest.raises(RetronMsdCompilerError, match="reverse_complement_fasta does not match"):
        _generate(repo_root=repo_root, materialized_root=materialized_root, out_dir=tmp_path / "outputs")


def _generate(*, repo_root: Path, materialized_root: Path, out_dir: Path) -> None:
    study_dir = repo_root / "docs" / "studies" / "retron_hairpin_design"
    generate_teto_pwm_trim_rescue_review_outputs(
        deliverable_plan_path=study_dir / "workbench" / "deliverables" / "teto_pwm_trim_rescue_v1.yaml",
        materialized_root=materialized_root,
        out_dir=out_dir,
        repo_root=repo_root,
        video_writer=fake_video_writer,
    )


def _sequence_index_rows(materialized_root: Path) -> list[dict[str, str]]:
    index_path = materialized_root / "manifest" / "indexes" / "sequence_index.tsv"
    return list(csv.DictReader(index_path.read_text(encoding="utf-8").splitlines(), delimiter="\t"))
