"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/permuter/tests/contracts/test_source_tree_contracts.py

Permuter source-tree information architecture contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dnadesign.permuter.src.core.registry import get_protocol
from dnadesign.permuter.src.workspaces.loader import load_workspace


def _permuter_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_python_module_headers_do_not_claim_agent_authorship() -> None:
    forbidden = ("Codex", "ChatGPT", "OpenAI", "Open AI")
    invalid: list[str] = []
    for path in sorted(_permuter_root().rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        header = "\n".join(path.read_text(encoding="utf-8").splitlines()[:12])
        if any(token in header for token in forbidden):
            invalid.append(path.relative_to(_permuter_root()).as_posix())

    assert invalid == []


def test_permuter_has_no_root_level_entrypoint_or_legacy_runtime_dirs() -> None:
    root = _permuter_root()

    forbidden = {
        "api.py",
        "cli.py",
        "jobs",
        "inputs",
        "results",
        "notebooks",
    }

    present = sorted(name for name in forbidden if (root / name).exists())
    assert present == []


def test_packaged_workspace_scopes_are_directory_scoped_configs() -> None:
    workspaces = _permuter_root() / "workspaces"
    config_paths = sorted(path for path in workspaces.glob("*/config.yaml") if path.parent.name != "_shared")

    assert config_paths
    for config_path in config_paths:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        assert sorted(payload) == ["scope"]
        assert payload["scope"]["name"] == config_path.parent.name
        assert payload["scope"]["output"]["dir"] == "${WORKSPACE_DIR}/outputs"
        assert "${JOB_DIR}" not in config_path.read_text(encoding="utf-8")
        assert load_workspace(config_path).scope_id == config_path.parent.name


def test_retired_rt_select_protocol_alias_fails_fast() -> None:
    with pytest.raises(ValueError, match="retired.*multisite_select"):
        get_protocol("rt_select")
