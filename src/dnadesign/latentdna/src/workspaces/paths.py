"""
Path and layout helpers for latentdna workspaces.
"""

from __future__ import annotations

from os import environ
from pathlib import Path

from ..contracts.errors import WorkspaceValidationError


def project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


def builtin_templates_dir() -> Path:
    return project_root() / "src" / "dnadesign" / "latentdna" / "workspaces" / "templates"


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    repo_candidate = project_root() / candidate
    if repo_candidate.exists():
        return repo_candidate.resolve()
    return (Path.cwd() / candidate).resolve()


def default_workspace_root() -> tuple[Path, str]:
    env_value = Path(Path.cwd().as_posix())
    source = "cwd"
    if environ.get("LATENTDNA_WORKSPACE_ROOT"):
        env_value = Path(environ["LATENTDNA_WORKSPACE_ROOT"])
        source = "env"
    return env_value.resolve(), source


def resolve_workspace_path(workspace: str | Path) -> Path:
    candidate = Path(workspace)
    if candidate.is_file():
        return candidate.parent.resolve()
    if candidate.is_dir():
        return candidate.resolve()
    cwd_candidate = Path.cwd() / candidate
    if cwd_candidate.is_dir():
        return cwd_candidate.resolve()
    raise WorkspaceValidationError(f"workspace not found: {workspace}")


def legacy_output_root(workspace_dir: Path) -> Path:
    return (workspace_dir / "outputs" / "latentdna").resolve()


def has_legacy_output_entries(path: Path) -> bool:
    if not path.exists():
        return False
    for candidate in path.rglob("*"):
        if candidate.name.startswith("."):
            continue
        return True
    return False
