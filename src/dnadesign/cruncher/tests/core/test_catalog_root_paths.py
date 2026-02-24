"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/core/test_catalog_root_paths.py

Validate catalog root path resolution contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dnadesign.cruncher.utils.paths import resolve_catalog_root


def _write_workspace_config(workspace_root: Path) -> Path:
    config_path = workspace_root / "configs" / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("cruncher:\n  schema_version: 3\n")
    return config_path


def test_relative_catalog_root_resolves_from_workspace_root(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    config_path = _write_workspace_config(workspace_root)

    resolved = resolve_catalog_root(config_path, ".cruncher/demo_pairwise")

    assert resolved == (workspace_root / ".cruncher" / "demo_pairwise").resolve()


def test_relative_catalog_root_cannot_escape_workspace_root(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    config_path = _write_workspace_config(workspace_root)

    with pytest.raises(ValueError, match="must stay within the workspace root"):
        resolve_catalog_root(config_path, "../shared-catalog")


def test_absolute_catalog_root_is_resolved_verbatim(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    config_path = _write_workspace_config(workspace_root)
    absolute_catalog_root = tmp_path / "catalog-root"

    resolved = resolve_catalog_root(config_path, absolute_catalog_root)

    assert resolved == absolute_catalog_root.resolve()
