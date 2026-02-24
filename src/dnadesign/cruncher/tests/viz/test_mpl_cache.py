"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/viz/test_mpl_cache.py

Validate Matplotlib cache contracts for Cruncher runtime commands.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dnadesign.cruncher.viz.mpl import ensure_mpl_cache


def test_ensure_mpl_cache_defaults_to_repo_shared_cache_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MPLCONFIGDIR", raising=False)
    resolved = ensure_mpl_cache(Path("/tmp/irrelevant-catalog-root"))
    expected = Path(__file__).resolve().parents[5] / ".cache" / "matplotlib" / "cruncher"
    assert resolved == expected
    assert resolved.exists()
    assert resolved.is_dir()


def test_ensure_mpl_cache_uses_env_override_when_set(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    target = tmp_path / "mpl-env"
    monkeypatch.setenv("MPLCONFIGDIR", str(target))
    resolved = ensure_mpl_cache(Path("/tmp/irrelevant-catalog-root"))
    assert resolved == target
    assert resolved.exists()
    assert resolved.is_dir()


def test_ensure_mpl_cache_fails_fast_when_repo_root_is_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MPLCONFIGDIR", raising=False)
    monkeypatch.setattr("dnadesign.cruncher.viz.mpl._repo_root_from", lambda _: None)
    with pytest.raises(RuntimeError, match="Unable to determine repository root for Matplotlib cache"):
        ensure_mpl_cache(Path("/tmp/irrelevant-catalog-root"))
