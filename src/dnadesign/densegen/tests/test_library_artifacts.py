"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/test_library_artifacts.py

Unit tests for library artifact persistence helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dnadesign.densegen.src.core.pipeline.library_artifacts import write_library_artifacts


def test_write_library_artifacts_raises_on_invalid_existing_parquet(tmp_path: Path) -> None:
    outputs_root = tmp_path / "outputs"
    libraries_dir = outputs_root / "libraries"
    libraries_dir.mkdir(parents=True)
    builds_path = libraries_dir / "library_builds.parquet"
    builds_path.write_text("not parquet", encoding="utf-8")

    with pytest.raises(Exception):
        write_library_artifacts(
            library_source="build",
            library_artifact=None,
            library_build_rows=[{"library_index": 1}],
            library_member_rows=[],
            outputs_root=outputs_root,
            cfg_path=tmp_path / "config.yaml",
            run_id="demo",
            run_root=tmp_path,
            config_hash="abc",
            pool_manifest_hash=None,
        )
