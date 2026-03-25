"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/tests/test_status_path_ref.py

Focused tests for the shared OPS path-reference contract.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dnadesign.ops.status.path_ref import resolve_path_ref


def test_path_ref_repo_base_resolution(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    target = repo_root / "usr" / "datasets"
    target.mkdir(parents=True)

    resolved = resolve_path_ref("usr/datasets", repo_root=repo_root, default_base="repo")

    assert resolved == target.resolve()


def test_path_ref_manifest_base_resolution(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "manifests"
    target = manifest_dir / "artifacts" / "latest.json"
    target.parent.mkdir(parents=True)

    resolved = resolve_path_ref("./artifacts/latest.json", manifest_dir=manifest_dir, default_base="manifest")

    assert resolved == target.resolve()


def test_path_ref_repo_escape_rejected(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True)

    with pytest.raises(ValueError, match="escapes repository root"):
        resolve_path_ref("repo:../outside.txt", repo_root=repo_root, default_base="repo")


def test_campaign_manifest_v2_path_base_repo(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    manifest_dir = repo_root / "docs" / "studies" / "promoter" / "demo"
    manifest_dir.mkdir(parents=True)
    target = repo_root / "src" / "dnadesign" / "usr" / "datasets"
    target.mkdir(parents=True)

    resolved = resolve_path_ref(
        "src/dnadesign/usr/datasets",
        repo_root=repo_root,
        manifest_dir=manifest_dir,
        default_base="repo",
    )

    assert resolved == target.resolve()
