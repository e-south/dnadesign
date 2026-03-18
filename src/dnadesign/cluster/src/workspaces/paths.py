"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cluster/src/workspaces/paths.py

Workspace path and scaffold helpers for cluster.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from ..layout import builtin_cluster_dir
from .errors import WorkspaceConfigError


def builtin_workspaces_dir() -> Path:
    return builtin_cluster_dir() / "workspaces"


def validate_workspace_id(workspace_id: str) -> str:
    value = str(workspace_id or "").strip()
    if not value:
        raise WorkspaceConfigError("workspace id must be a non-empty string")
    if value in {".", ".."} or any(ch in value for ch in ("/", "\\")):
        raise WorkspaceConfigError("workspace id must be a simple directory name (not a path)")
    return value


def render_workspace_template(workspace_id: str) -> str:
    workspace_name = validate_workspace_id(workspace_id)
    return dedent(
        f"""\
        schema_version: 1
        input:
          # Set exactly one input source before running cluster.
          # dataset: "my_usr_dataset"
          # file: "/absolute/path/to/records.parquet"

        fit:
          name: "{workspace_name}"
          x_col: "replace_me"
          method: "leiden"
          write: false

        umap:
          name: "{workspace_name}"
          x_col: "replace_me"
          write: false

        analyze:
          cluster_col: "cluster__{workspace_name}"
          group_by: "source"
        """
    )


def resolve_workspace_dir(workspace: str | Path) -> Path:
    spec = Path(workspace).expanduser()
    candidates: list[Path] = []
    if spec.is_absolute():
        candidates.append(spec)
    elif len(spec.parts) > 1 or "/" in str(workspace) or str(workspace).startswith("."):
        candidates.append((Path.cwd() / spec).resolve())
    else:
        candidates.append((builtin_workspaces_dir() / str(workspace)).resolve())
        candidates.append((Path.cwd() / str(workspace)).resolve())

    tried: list[str] = []
    for candidate in candidates:
        if candidate.is_file() and candidate.name == "config.yaml":
            tried.append(str(candidate))
            return candidate.parent.resolve()
        tried.append(str(candidate / "config.yaml"))
        if candidate.is_dir() and (candidate / "config.yaml").is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Workspace config not found. Pass a packaged workspace id or a directory containing config.yaml.\n"
        + "\n".join(f"  - {item}" for item in tried)
    )


def list_builtin_workspaces() -> list[str]:
    root = builtin_workspaces_dir()
    if not root.exists():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir() and (path / "config.yaml").is_file())


def init_workspace(*, workspace_id: str, root: str | Path | None = None) -> Path:
    workspace_name = validate_workspace_id(workspace_id)
    root_path = Path.cwd() if root is None else Path(root).expanduser().resolve()
    if root_path.exists() and not root_path.is_dir():
        raise WorkspaceConfigError(f"Workspace root is not a directory: {root_path}")
    root_path.mkdir(parents=True, exist_ok=True)
    workspace_dir = (root_path / workspace_name).resolve()
    if workspace_dir.exists():
        raise WorkspaceConfigError(f"Workspace directory already exists: {workspace_dir}")
    workspace_dir.mkdir(parents=False, exist_ok=False)
    (workspace_dir / "outputs" / "cluster").mkdir(parents=True, exist_ok=True)
    (workspace_dir / "config.yaml").write_text(render_workspace_template(workspace_name), encoding="utf-8")
    return workspace_dir


__all__ = [
    "builtin_workspaces_dir",
    "init_workspace",
    "list_builtin_workspaces",
    "render_workspace_template",
    "resolve_workspace_dir",
    "validate_workspace_id",
]
