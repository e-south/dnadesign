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

from dnadesign.permuter.src.core.config import ScopeConfig
from dnadesign.permuter.src.core.paths import CONFIG_NAME, expand_for_workspace
from dnadesign.permuter.src.workspaces.contracts import PermuterWorkspace


def load_workspace(workspace: Path | str) -> PermuterWorkspace:
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
        config = ScopeConfig.model_validate(data)
    except Exception as exc:
        raise ValueError(f"workspace config does not satisfy Permuter config contract: {config_path}: {exc}") from exc
    workspace_root = config_path.parent
    scope_id = workspace_root.name
    if config.scope.name != scope_id:
        raise ValueError(
            f"workspace scope id must match scope.name: scope={scope_id!r} scope.name={config.scope.name!r}"
        )
    output_root = expand_for_workspace(config.scope.output.dir, workspace_dir=workspace_root)
    try:
        output_root.relative_to(workspace_root)
    except ValueError as exc:
        raise ValueError(
            f"workspace output.dir must resolve inside the workspace root: {config.scope.output.dir!r} -> {output_root}"
        ) from exc
    return PermuterWorkspace(
        scope_id=scope_id,
        root=workspace_root,
        config_path=config_path,
        config=config,
    )


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
