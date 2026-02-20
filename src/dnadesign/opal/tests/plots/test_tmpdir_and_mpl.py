"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/plots/test_tmpdir_and_mpl.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from pathlib import Path


def test_resolve_opal_tmpdir_prefers_workdir(tmp_path: Path) -> None:
    from dnadesign.opal.src.core.tmpdir import resolve_opal_tmpdir

    workdir = tmp_path / "campaign"
    path = resolve_opal_tmpdir(workdir=workdir)
    assert path.exists()
    assert path.is_dir()
    assert path.parts[-2:] == (".opal", "tmp")


def test_resolve_opal_tmpdir_env_override(tmp_path: Path, monkeypatch) -> None:
    from dnadesign.opal.src.core.tmpdir import resolve_opal_tmpdir

    override = tmp_path / "opal_override"
    monkeypatch.setenv("OPAL_TMPDIR", str(override))
    path = resolve_opal_tmpdir()
    assert path == override
    assert path.exists()


def test_ensure_mpl_config_dir_sets_env(tmp_path: Path, monkeypatch) -> None:
    from dnadesign.opal.src.plots._mpl_utils import ensure_mpl_config_dir

    monkeypatch.delenv("MPLCONFIGDIR", raising=False)
    workdir = tmp_path / "campaign"
    path = ensure_mpl_config_dir(workdir=workdir)
    assert Path(os.environ["MPLCONFIGDIR"]) == path
    assert path.exists()
    assert path.name == "mpl"
