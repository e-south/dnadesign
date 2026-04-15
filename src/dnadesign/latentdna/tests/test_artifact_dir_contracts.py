from __future__ import annotations

from pathlib import Path

import pytest

from dnadesign.latentdna.src.contracts.errors import ArtifactConflictError
from dnadesign.latentdna.src.io.artifact_dirs import commit_staged_artifact_dirs


def test_commit_staged_artifact_dirs_reports_concurrent_conflict(tmp_path, monkeypatch) -> None:
    staging_dir = tmp_path / "staging"
    final_dir = tmp_path / "final"
    staging_dir.mkdir()
    final_dir.mkdir()

    original_exists = Path.exists
    original_rename = Path.rename
    rename_attempted = False

    def fake_exists(self: Path) -> bool:
        nonlocal rename_attempted
        if self == final_dir and not rename_attempted:
            return False
        return original_exists(self)

    def fake_rename(self: Path, target: Path | str) -> Path:
        nonlocal rename_attempted
        if self == staging_dir and Path(target) == final_dir:
            rename_attempted = True
            raise OSError(66, "Directory not empty")
        return original_rename(self, target)

    monkeypatch.setattr(Path, "exists", fake_exists)
    monkeypatch.setattr(Path, "rename", fake_rename)

    with pytest.raises(ArtifactConflictError, match="serialize concurrent runs"):
        commit_staged_artifact_dirs([(staging_dir, final_dir)], force=False)
