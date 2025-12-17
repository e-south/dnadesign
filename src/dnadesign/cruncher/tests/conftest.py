"""
--------------------------------------------------------------------------------
<dnadesign project>
cruncher/tests/conftest.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dnadesign.cruncher.utils.config import load_config

_DATA_DIR = Path(__file__).resolve().parent / "data"


def _write_yaml(d: dict, path: Path) -> None:
    path.write_text(yaml.safe_dump(d, sort_keys=False))


# fixtures
@pytest.fixture
def mini_cfg_path(tmp_path: Path) -> Path:
    """
    Minimal modern cruncher config for tests, using only bundled MEME files.
    """
    cfg_dict = {
        "cruncher": {
            "mode": "parse",
            "out_dir": str(tmp_path / "results"),
            "regulator_sets": [["lexA", "oxyR"]],
            "parse": {
                "motif_root": str(_DATA_DIR),
                "formats": {".txt": "MEME"},
                "plot": {"logo": True, "bits_mode": "information", "dpi": 80},
            },
        }
    }

    dst = tmp_path / "example_test.yaml"
    _write_yaml(cfg_dict, dst)
    return dst  # tests hand this path to CLI or loader


@pytest.fixture
def cfg_obj(mini_cfg_path: Path):
    """
    Parses the YAML into *CruncherConfig* (Pydantic model) so
    unit tests can interact directly with strongly-typed config.
    """
    return load_config(mini_cfg_path)


@pytest.fixture
def cwd_tmp(tmp_path: Path, monkeypatch):
    """
    Ensure CLI tests have an isolated CWD.
    """
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture(params=["oxyR.txt", "lexA.txt"])
def meme_file(request) -> Path:
    """
    Return a Path to one of the real MEME motif files
    located in cruncher/tests/data/.
    """
    return _DATA_DIR / request.param
