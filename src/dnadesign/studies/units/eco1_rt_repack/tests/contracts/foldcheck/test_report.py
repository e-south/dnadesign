"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/contracts/foldcheck/test_report.py

Eco1 fold-check report contract tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shutil
from pathlib import Path

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.suite import validate_checked_in_contracts
from dnadesign.studies.units.eco1_rt_repack.paths import DEFAULT_THREAD_OUTPUT_ROOT
from dnadesign.studies.units.eco1_rt_repack.tests._helpers import repo_root


def test_phase3_requires_materialized_foldcheck_report_when_phase2_artifacts_exist(tmp_path: Path) -> None:
    root = repo_root()
    source_output_root = root / DEFAULT_THREAD_OUTPUT_ROOT
    output_root = tmp_path / "thread-output"
    shutil.copytree(source_output_root, output_root, ignore=shutil.ignore_patterns("foldcheck_report.parquet"))

    report = validate_checked_in_contracts(
        repo_root=root,
        phase="phase3_foldcheck_report",
        output_root=output_root,
    )

    assert report.passed is False
    assert {issue.check_id for issue in report.issues} == {"eco1_rt.foldcheck_report.not_materialized"}
