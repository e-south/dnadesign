"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/workspace.py

Workspace discovery, scaffolding, and strict workspace job-path resolution.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from .core import SchemaError, ensure

_WORKSPACE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class Workspace:
    name: str
    root: Path
    job_path: Path
    inputs_dir: Path
    outputs_dir: Path


def default_workspaces_root(*, base_root: Path | None = None) -> Path:
    root = Path.cwd() if base_root is None else Path(base_root).expanduser()
    return root / "workspaces"


def _validate_workspace_name(name: str) -> str:
    raw = str(name).strip()
    ensure(raw != "", "workspace name must be a non-empty string", SchemaError)
    ensure(_WORKSPACE_NAME_RE.fullmatch(raw) is not None, f"invalid workspace name: {raw!r}", SchemaError)
    return raw


def _resolve_root(root: Path | None) -> Path:
    if root is None:
        return default_workspaces_root().resolve()
    return Path(root).expanduser().resolve()


def workspace_root(name: str, *, root: Path | None = None) -> Path:
    workspace_name = _validate_workspace_name(name)
    return _resolve_root(root) / workspace_name


def workspace_job_path(name: str, *, root: Path | None = None) -> Path:
    ws_root = workspace_root(name, root=root)
    job_path = ws_root / "job.yaml"
    if not job_path.exists():
        raise SchemaError(f"workspace '{name}' does not contain job.yaml: {job_path}")
    return job_path


def resolve_workspace_job_path(name: str, *, root: Path | None = None) -> Path:
    return workspace_job_path(name, root=root)


def discover_workspaces(*, root: Path | None = None) -> tuple[Workspace, ...]:
    root_path = _resolve_root(root)
    if not root_path.exists():
        return ()
    if not root_path.is_dir():
        raise SchemaError(f"workspaces root is not a directory: {root_path}")

    out: list[Workspace] = []
    for child in sorted(root_path.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        if child.name.startswith("_"):
            continue
        job_path = child / "job.yaml"
        if not job_path.exists():
            continue
        out.append(
            Workspace(
                name=child.name,
                root=child.resolve(),
                job_path=job_path.resolve(),
                inputs_dir=(child / "inputs").resolve(),
                outputs_dir=(child / "outputs").resolve(),
            )
        )
    return tuple(out)


def _workspace_job_template() -> dict:
    return {
        "version": 3,
        "results_root": "outputs",
        "input": {
            "kind": "parquet",
            "path": "inputs/input.parquet",
            "adapter": {
                "kind": "generic_features",
                "columns": {
                    "sequence": "sequence",
                    "features": "features",
                    "effects": "effects",
                    "display": "display",
                    "id": "id",
                },
                "policies": {},
            },
            "alphabet": "DNA",
        },
        "render": {"renderer": "sequence_rows", "style": {"preset": "presentation_default", "overrides": {}}},
        "outputs": [{"kind": "images", "dir": "plots", "fmt": "png"}],
        "run": {"strict": False, "fail_on_skips": False, "emit_report": True},
    }


def init_workspace(name: str, *, root: Path | None = None) -> Workspace:
    ws_root = workspace_root(name, root=root)
    root_path = ws_root.parent
    if root_path.exists() and not root_path.is_dir():
        raise SchemaError(f"workspaces root is not a directory: {root_path}")
    root_path.mkdir(parents=True, exist_ok=True)

    if ws_root.exists():
        raise SchemaError(f"workspace already exists: {ws_root}")

    inputs_dir = ws_root / "inputs"
    outputs_dir = ws_root / "outputs"
    reports_dir = ws_root / "reports"
    job_path = ws_root / "job.yaml"

    ws_root.mkdir(parents=False, exist_ok=False)
    inputs_dir.mkdir(parents=False, exist_ok=False)
    outputs_dir.mkdir(parents=False, exist_ok=False)
    reports_dir.mkdir(parents=False, exist_ok=False)

    job_path.write_text(yaml.safe_dump(_workspace_job_template(), sort_keys=False))

    return Workspace(
        name=ws_root.name,
        root=ws_root.resolve(),
        job_path=job_path.resolve(),
        inputs_dir=inputs_dir.resolve(),
        outputs_dir=outputs_dir.resolve(),
    )
