"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/workspaces/loader.py

Workspace discovery and config loading.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from dnadesign.permuter.src.workspaces.contracts import WorkspaceConfig

CONFIG_NAME = "config.yaml"


def load_workspace(workspace: Path | str) -> WorkspaceConfig:
    root = Path(workspace).expanduser().resolve()
    config_path = root / CONFIG_NAME if root.is_dir() else root
    if config_path.name != CONFIG_NAME:
        raise ValueError(f"workspace config must be named {CONFIG_NAME!r}; got {config_path}")
    if not config_path.exists():
        raise ValueError(f"workspace config not found: {config_path}")
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"workspace config is not valid YAML: {config_path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"workspace config must be a mapping: {config_path}")
    try:
        return WorkspaceConfig.model_validate(data)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc


def find_workspaces(root: Path | str) -> list[Path]:
    base = Path(root).expanduser().resolve()
    if not base.exists():
        raise ValueError(f"workspace root not found: {base}")
    if base.is_file():
        return [base] if base.name == CONFIG_NAME else []
    direct = base / CONFIG_NAME
    if direct.exists():
        return [direct]
    return sorted(path for path in base.glob(f"*/{CONFIG_NAME}") if path.is_file())
