"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/contracts/foldcheck/test_request.py

Eco1 fold-check request contract tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.foldcheck import (
    validate_foldcheck_request_content,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.foldcheck_request import (
    materialize_foldcheck_request,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.foldcheck_request._fixtures import (
    write_minimal_foldcheck_inputs,
)


def test_foldcheck_request_contract_accepts_materialized_request(tmp_path: Path) -> None:
    write_minimal_foldcheck_inputs(tmp_path)
    result = materialize_foldcheck_request(repo_root=Path.cwd(), output_root=tmp_path)

    issues = validate_foldcheck_request_content(result.request_manifest_path, output_root=tmp_path)

    assert issues == []


def test_foldcheck_request_contract_resolves_manifest_relative_fasta_path(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    write_minimal_foldcheck_inputs(source_root)
    result = materialize_foldcheck_request(repo_root=Path.cwd(), output_root=source_root)
    deployed_root = tmp_path / "deployed"
    shutil.copytree(source_root, deployed_root)
    deployed_manifest = deployed_root / result.request_manifest_path.relative_to(source_root)

    issues = validate_foldcheck_request_content(deployed_manifest, output_root=deployed_root)

    assert issues == []


def test_foldcheck_request_contract_rejects_stale_upstream_hash(tmp_path: Path) -> None:
    write_minimal_foldcheck_inputs(tmp_path)
    result = materialize_foldcheck_request(repo_root=Path.cwd(), output_root=tmp_path)
    manifest = yaml.safe_load(result.request_manifest_path.read_text(encoding="utf-8"))
    manifest["upstream_artifact_hashes"]["candidate_table"] = "sha256:" + "0" * 64
    result.request_manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    issues = validate_foldcheck_request_content(result.request_manifest_path, output_root=tmp_path)

    assert "eco1_rt.foldcheck_request.request_hash_mismatch" in {issue.check_id for issue in issues}
    assert "eco1_rt.foldcheck_request.upstream_hash_mismatch" in {issue.check_id for issue in issues}
