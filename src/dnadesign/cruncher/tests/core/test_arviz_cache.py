"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/core/test_arviz_cache.py

Validate ArviZ cache-directory setup for writable runtime diagnostics.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os

from dnadesign.cruncher.utils.arviz_cache import ensure_arviz_data_dir


def test_ensure_arviz_data_dir_sets_env_under_catalog_root(tmp_path) -> None:
    catalog_root = tmp_path / "catalog"
    os.environ.pop("ARVIZ_DATA", None)

    cache_dir = ensure_arviz_data_dir(catalog_root)
    arviz_data_dir = catalog_root / ".arviz_data"

    assert cache_dir.name == "arviz_data"
    assert cache_dir.is_dir()
    assert os.environ.get("ARVIZ_DATA") == str(arviz_data_dir)
    assert arviz_data_dir.is_dir()


def test_ensure_arviz_data_dir_overrides_existing_env(tmp_path) -> None:
    custom = tmp_path / "custom_arviz"
    os.environ["ARVIZ_DATA"] = str(custom)

    cache_dir = ensure_arviz_data_dir(tmp_path / "catalog")
    arviz_data_dir = tmp_path / "catalog" / ".arviz_data"

    assert cache_dir.name == "arviz_data"
    assert cache_dir.is_dir()
    assert os.environ.get("ARVIZ_DATA") == str(arviz_data_dir)
    assert arviz_data_dir.is_dir()


def test_ensure_arviz_data_dir_overrides_unwritable_home(tmp_path) -> None:
    blocked_home = tmp_path / "blocked-home"
    blocked_home.write_text("x")
    os.environ["HOME"] = str(blocked_home)

    cache_dir = ensure_arviz_data_dir(tmp_path / "catalog")

    assert cache_dir == (tmp_path / "catalog" / ".runtime_home" / "arviz_data")
    assert cache_dir.is_dir()
    assert os.environ.get("HOME") == str(tmp_path / "catalog" / ".runtime_home")
