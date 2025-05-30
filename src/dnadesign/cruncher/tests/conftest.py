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

from dnadesign.cruncher.config import load_config

_DATA_DIR = Path(__file__).with_suffix("").parent / "data"


# helpers
def _load_yaml_to_dict(rel_path: Path) -> dict:
    "Read YAML file bundled with tests and return mutable dict."
    txt = rel_path.read_text()
    return yaml.safe_load(txt)


def _write_yaml(d: dict, path: Path) -> None:
    path.write_text(yaml.safe_dump(d, sort_keys=False))


# fixtures
@pytest.fixture
def mini_cfg_path(tmp_path: Path) -> Path:
    """
    Return a **Path** to a *modified* YAML ready for tests.

    - Starts from repo-root configs/example.yaml
    - Shrinks draws/chains so sampling tests finish fast
    - Redirects out_dir into the tmp_path sandbox
    """
    repo_cfg = Path(__file__).resolve().parents[2] / "configs" / "example.yaml"
    cfg_dict = _load_yaml_to_dict(repo_cfg)

    # Patch for speed & isolation
    cfg = cfg_dict["cruncher"]
    cfg["out_dir"] = str(tmp_path / "results")
    if cfg.get("sample"):
        gibbs = cfg["sample"]["optimiser"]["gibbs"]
        gibbs["draws"] = 5
        gibbs["chains"] = 1

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
