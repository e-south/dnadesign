"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/permuter/workspace.py

Workspace loader: parses workspace.yaml and returns an experiments root plus
a list of experiment references. Relative paths in workspace.yaml are resolved
relative to the workspace.yaml.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
import os
import yaml


class WorkspaceError(RuntimeError):
    """Raised when a workspace.yaml file is malformed or inconsistent."""


@dataclass(frozen=True)
class ExperimentRef:
    """
    A single experiment referenced from workspace.yaml

    Attributes:
        name: Logical experiment name (unique within workspace).
        dir: The experiment's directory (usually contains config.yaml and inputs/).
        config_path: The path to the experiment's config.yaml.
        enabled: Whether this experiment should be run by default.
    """

    name: str
    dir: Path
    config_path: Path
    enabled: bool = True


def is_workspace_config(path: Path) -> bool:
    """
    Best-effort check whether a YAML file is a workspace.yaml (not an experiment config).
    Safe: returns False on any parse error.
    """
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return bool(data and "permuter" in data and "workspace" in data["permuter"])
    except Exception:
        return False


def _resolve_relative_to(base_dir: Path, maybe_path: str | Path) -> Path:
    """
    Resolve a user-provided path relative to base_dir when not absolute.
    Expands ~ and environment variables first.
    """
    raw = os.path.expandvars(str(maybe_path))
    p = Path(raw).expanduser()
    return p if p.is_absolute() else (base_dir / p)


def load_workspace(path: Path) -> Tuple[Path, List[ExperimentRef]]:
    """
    Parse a workspace.yaml and return (experiments_root, list_of_experiments).

    Resolution rules:
      - experiments_dir is resolved relative to the directory containing workspace.yaml
        (unless it is absolute).
      - Each run's `config` is resolved relative to its experiment directory
        experiments_dir/<name>/ (unless absolute).
    """
    try:
        ws_text = path.read_text(encoding="utf-8")
        data = yaml.safe_load(ws_text)
        ws = data["permuter"]["workspace"]
    except Exception as e:  # pragma: no cover
        raise WorkspaceError(f"Invalid workspace.yaml: {e}")

    ws_dir = path.parent.resolve()

    # Resolve experiments_dir relative to the workspace.yaml location
    experiments_dir_raw = ws.get("experiments_dir", "dnadesign/permuter/experiments")
    exp_dir = _resolve_relative_to(ws_dir, experiments_dir_raw).resolve()
    if not exp_dir.is_dir():
        raise WorkspaceError(f"experiments_dir not found: {exp_dir}")

    runs = ws.get("runs", [])
    if not isinstance(runs, list) or not runs:
        raise WorkspaceError("workspace.runs must be a non-empty list")

    exps: List[ExperimentRef] = []
    seen: set[str] = set()

    for r in runs:
        name = str(r.get("name", "")).strip()
        if not name:
            raise WorkspaceError("Each run must have a 'name'")
        if name in seen:
            raise WorkspaceError(f"Duplicate experiment name: {name}")
        seen.add(name)

        # experiments/<name> directory
        exp_dir_i = (exp_dir / name).resolve()
        if not exp_dir_i.is_dir():
            raise WorkspaceError(f"{name}: experiment directory not found at {exp_dir_i}")

        # Resolve config path relative to experiments/<name>/ unless absolute
        cfg_rel = r.get("config", "config.yaml")
        cfg_path = _resolve_relative_to(exp_dir_i, cfg_rel).resolve()
        if not cfg_path.is_file():
            raise WorkspaceError(f"{name}: config not found at {cfg_path}")

        exps.append(
            ExperimentRef(
                name=name,
                dir=exp_dir_i,
                config_path=cfg_path,
                enabled=bool(r.get("enabled", True)),
            )
        )

    return exp_dir, exps


def iter_enabled(
    experiments: List[ExperimentRef], only: Optional[List[str]] = None
) -> Iterable[ExperimentRef]:
    """
    Yield enabled experiments, optionally filtered by names in `only`.
    """
    if only:
        only_set = {n.strip() for n in only}
        for e in experiments:
            if e.name in only_set:
                yield e
    else:
        for e in experiments:
            if e.enabled:
                yield e
