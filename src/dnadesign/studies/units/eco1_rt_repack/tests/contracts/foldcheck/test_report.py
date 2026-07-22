"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/contracts/foldcheck/test_report.py

Eco1 fold-check report contract tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.sampling import artifacts as sampling_artifacts
from dnadesign.studies.units.eco1_rt_repack.operations.contracts.sampling.artifacts import (
    validate_sampling_artifacts,
)
from dnadesign.studies.units.eco1_rt_repack.tests._helpers import repo_root


def test_phase3_requires_materialized_foldcheck_report_when_phase2_artifacts_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _accept_phase2_artifact_content(monkeypatch)
    _write_phase2_artifact_placeholders(tmp_path)

    issues = validate_sampling_artifacts(
        repo_root=repo_root(),
        structure_root=tmp_path,
        phase="phase3_foldcheck_report",
    )

    assert [issue.check_id for issue in issues] == ["eco1_rt.foldcheck_report.not_materialized"]


def _accept_phase2_artifact_content(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "validate_thread_plan_content",
        "validate_proteinmpnn_request_content",
        "validate_sample_table_content",
        "validate_candidate_table_content",
        "validate_foldcheck_request_content",
    ):
        monkeypatch.setattr(sampling_artifacts, name, lambda *args, **kwargs: [])


def _write_phase2_artifact_placeholders(output_root: Path) -> None:
    for relative_path in (
        "thread_plan.yaml",
        "proteinmpnn_request/request_manifest.yaml",
        "sample_table.parquet",
        "candidate_table.parquet",
        "foldcheck_request/foldcheck_request_manifest.yaml",
    ):
        path = output_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture: true\n", encoding="utf-8")
