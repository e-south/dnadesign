"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/src/workspace.py

Workspace root, template, and scaffold contracts for infer CLI workflows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

from .errors import ConfigError

_WORKSPACE_TEMPLATE_BY_PROFILE = {
    "local": "workspace_local_records_config.yaml",
    "usr-pressure": "pressure_test_infer_config.yaml",
}


def _repo_root_from(start: Path) -> Path | None:
    try:
        cursor = start.resolve()
    except Exception:
        cursor = start
    for root in [cursor, *cursor.parents]:
        if (root / "pyproject.toml").exists() or (root / ".git").exists():
            return root
    return None


def _infer_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_directory_path(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists() and not resolved.is_dir():
        raise ConfigError(f"{label} is not a directory: {resolved}")
    return resolved


def normalize_workspace_id(workspace_id: str) -> str:
    value = str(workspace_id or "").strip()
    if not value:
        raise ConfigError("workspace id must be a non-empty string")
    if value in {".", ".."} or any(ch in value for ch in ("/", "\\")):
        raise ConfigError("workspace id must be a simple directory name (not a path)")
    return value


def resolve_workspace_root(root: Optional[Path]) -> tuple[Path, str]:
    if root is not None:
        return _ensure_directory_path(root, label="workspace root"), "arg"

    env_root = str(os.environ.get("INFER_WORKSPACE_ROOT") or "").strip()
    if env_root:
        return _ensure_directory_path(Path(env_root), label="INFER_WORKSPACE_ROOT"), "env"

    repo_root = _repo_root_from(Path.cwd())
    if repo_root is None:
        raise ConfigError(
            "Unable to determine workspace root. Pass --root, set INFER_WORKSPACE_ROOT, "
            "or run from inside the dnadesign repository."
        )
    return _ensure_directory_path(repo_root / "src" / "dnadesign" / "infer" / "workspaces", label="workspace root"), (
        "repo-default"
    )


def resolve_workspace_template(template: Optional[Path], *, profile: str = "local") -> Path:
    if template is not None:
        resolved = template.expanduser().resolve()
        if not resolved.exists() or not resolved.is_file():
            raise ConfigError(f"template config file not found: {resolved}")
        return resolved

    profile_key = str(profile or "").strip()
    template_name = _WORKSPACE_TEMPLATE_BY_PROFILE.get(profile_key)
    if template_name is None:
        choices = ", ".join(sorted(_WORKSPACE_TEMPLATE_BY_PROFILE))
        raise ConfigError(f"workspace profile must be one of: {choices}")

    resolved = (_infer_root() / "docs" / "operations" / "examples" / template_name).resolve()
    if not resolved.exists() or not resolved.is_file():
        raise ConfigError(
            "default workspace template not found: "
            f"{resolved}. Pass --template with an explicit config path."
        )
    return resolved


def init_workspace(*, workspace_id: str, root: Optional[Path], template: Optional[Path], profile: str = "local") -> Path:
    workspace_name = normalize_workspace_id(workspace_id)
    root_path, _source = resolve_workspace_root(root)
    template_path = resolve_workspace_template(template, profile=profile)

    root_path.mkdir(parents=True, exist_ok=True)
    workspace_dir = (root_path / workspace_name).resolve()
    if workspace_dir.exists():
        raise ConfigError(f"workspace already exists: {workspace_dir}")

    workspace_dir.mkdir(parents=False, exist_ok=False)
    (workspace_dir / "inputs").mkdir(parents=True, exist_ok=True)
    (workspace_dir / "outputs" / "logs" / "ops" / "audit").mkdir(parents=True, exist_ok=True)
    if profile == "usr-pressure":
        (workspace_dir / "outputs" / "usr_datasets").mkdir(parents=True, exist_ok=True)
    shutil.copy2(template_path, workspace_dir / "config.yaml")
    return workspace_dir
