"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cluster/src/workspaces/paths.py

Workspace path and scaffold helpers for cluster.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from datetime import datetime
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
        candidates.append((Path.cwd() / str(workspace)).resolve())
        candidates.append((builtin_workspaces_dir() / str(workspace)).resolve())

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
    return [str(entry["workspace_id"]) for entry in list_builtin_workspace_inventory()]


def _workspace_inventory_entry(*, workspace_dir: Path) -> dict[str, object]:
    outputs_dir = workspace_dir / "outputs"
    output_files = 0
    latest_output_timestamp: float | None = None
    if outputs_dir.exists():
        for candidate in outputs_dir.rglob("*"):
            if not candidate.is_file():
                continue
            output_files += 1
            try:
                stat_result = candidate.stat()
            except OSError:
                continue
            if latest_output_timestamp is None or stat_result.st_mtime > latest_output_timestamp:
                latest_output_timestamp = stat_result.st_mtime
    latest_output_mtime = (
        datetime.fromtimestamp(latest_output_timestamp).astimezone().isoformat(timespec="seconds")
        if latest_output_timestamp is not None
        else None
    )
    return {
        "workspace_id": workspace_dir.name,
        "workspace_dir": str(workspace_dir.resolve()),
        "workspace_state": "attention" if output_files else "clean",
        "output_files": output_files,
        "latest_output_mtime": latest_output_mtime,
    }


def list_builtin_workspace_inventory() -> list[dict[str, object]]:
    root = builtin_workspaces_dir()
    if not root.exists():
        return []
    inventory: list[dict[str, object]] = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        if not (path / "config.yaml").is_file():
            continue
        inventory.append(_workspace_inventory_entry(workspace_dir=path))
    return inventory


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
    "list_builtin_workspace_inventory",
    "list_builtin_workspaces",
    "render_workspace_template",
    "resolve_workspace_dir",
    "validate_workspace_id",
]
