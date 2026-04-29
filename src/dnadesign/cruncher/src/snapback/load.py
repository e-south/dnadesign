"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/snapback/load.py

Load v2 explicit and v3 co-design solve snapback specs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from dnadesign.cruncher.nickases.catalog import resolve_workspace_relative_path
from dnadesign.cruncher.snapback.errors import SnapbackSpecError
from dnadesign.cruncher.snapback.models import SingleNickSnapbackSpec
from dnadesign.cruncher.snapback.released_models import SingleNickReleasedSnapbackSpec
from dnadesign.cruncher.snapback.solve_models import SingleNickSnapbackSolveSpec
from dnadesign.cruncher.snapback.visual_models import SingleNickSnapbackVisualSpec


def _load_yaml_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise SnapbackSpecError(f"Invalid YAML in {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SnapbackSpecError(f"{label} {path} must be a YAML mapping.")
    return payload


def resolve_workspace_root_for_snapback_spec(spec_path: Path) -> Path:
    resolved = spec_path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Snapback spec not found: {resolved}")
    if not resolved.name.endswith(".snapback.yaml") or resolved.name.endswith(".snapback.solve.yaml"):
        raise SnapbackSpecError("--spec must point to a <workspace>/configs/snapback/<name>.snapback.yaml file.")
    if len(resolved.parents) < 3 or resolved.parent.name != "snapback" or resolved.parent.parent.name != "configs":
        raise SnapbackSpecError("--spec must point to a <workspace>/configs/snapback/<name>.snapback.yaml file.")
    return resolved.parent.parent.parent.resolve()


def resolve_workspace_root_for_snapback_solve_spec(spec_path: Path) -> Path:
    resolved = spec_path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Snapback solve spec not found: {resolved}")
    if not resolved.name.endswith(".snapback.solve.yaml"):
        raise SnapbackSpecError("--spec must point to a <workspace>/configs/snapback/<name>.snapback.solve.yaml file.")
    if len(resolved.parents) < 3 or resolved.parent.name != "snapback" or resolved.parent.parent.name != "configs":
        raise SnapbackSpecError("--spec must point to a <workspace>/configs/snapback/<name>.snapback.solve.yaml file.")
    return resolved.parent.parent.parent.resolve()


def resolve_workspace_root_for_released_snapback_spec(spec_path: Path) -> Path:
    resolved = spec_path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Released-product snapback spec not found: {resolved}")
    if not resolved.name.endswith(".released.snapback.yaml"):
        raise SnapbackSpecError(
            "--spec must point to a <workspace>/configs/snapback/<name>.released.snapback.yaml file."
        )
    if len(resolved.parents) < 3 or resolved.parent.name != "snapback" or resolved.parent.parent.name != "configs":
        raise SnapbackSpecError(
            "--spec must point to a <workspace>/configs/snapback/<name>.released.snapback.yaml file."
        )
    return resolved.parent.parent.parent.resolve()


def resolve_workspace_root_for_snapback_visual_spec(spec_path: Path) -> Path:
    resolved = spec_path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Snapback visual spec not found: {resolved}")
    if not resolved.name.endswith(".visual.snapback.yaml"):
        raise SnapbackSpecError("--spec must point to a <workspace>/configs/snapback/<name>.visual.snapback.yaml file.")
    if len(resolved.parents) < 3 or resolved.parent.name != "snapback" or resolved.parent.parent.name != "configs":
        raise SnapbackSpecError("--spec must point to a <workspace>/configs/snapback/<name>.visual.snapback.yaml file.")
    return resolved.parent.parent.parent.resolve()


def load_snapback_spec(path: str | Path) -> tuple[SingleNickSnapbackSpec, Path, Path]:
    spec_path = Path(path).expanduser().resolve()
    workspace_root = resolve_workspace_root_for_snapback_spec(spec_path)
    payload = _load_yaml_mapping(spec_path, label="snapback spec")
    try:
        document = SingleNickSnapbackSpec.model_validate(payload)
    except Exception as exc:
        raise SnapbackSpecError(f"Snapback schema validation failed for {spec_path}: {exc}") from exc
    return document, spec_path, workspace_root


def load_snapback_visual_spec(path: str | Path) -> tuple[SingleNickSnapbackVisualSpec, Path, Path]:
    spec_path = Path(path).expanduser().resolve()
    workspace_root = resolve_workspace_root_for_snapback_visual_spec(spec_path)
    payload = _load_yaml_mapping(spec_path, label="snapback visual spec")
    try:
        document = SingleNickSnapbackVisualSpec.model_validate(payload)
    except Exception as exc:
        raise SnapbackSpecError(f"Snapback visual schema validation failed for {spec_path}: {exc}") from exc
    return document, spec_path, workspace_root


def load_snapback_solve_spec(path: str | Path) -> tuple[SingleNickSnapbackSolveSpec, Path, Path]:
    spec_path = Path(path).expanduser().resolve()
    workspace_root = resolve_workspace_root_for_snapback_solve_spec(spec_path)
    payload = _load_yaml_mapping(spec_path, label="snapback solve spec")
    try:
        document = SingleNickSnapbackSolveSpec.model_validate(payload)
    except Exception as exc:
        raise SnapbackSpecError(f"Snapback solve schema validation failed for {spec_path}: {exc}") from exc
    return document, spec_path, workspace_root


def load_released_snapback_spec(path: str | Path) -> tuple[SingleNickReleasedSnapbackSpec, Path, Path]:
    spec_path = Path(path).expanduser().resolve()
    workspace_root = resolve_workspace_root_for_released_snapback_spec(spec_path)
    payload = _load_yaml_mapping(spec_path, label="released-product snapback spec")
    try:
        document = SingleNickReleasedSnapbackSpec.model_validate(payload)
    except Exception as exc:
        raise SnapbackSpecError(f"Released-product snapback schema validation failed for {spec_path}: {exc}") from exc
    return document, spec_path, workspace_root


__all__ = [
    "load_released_snapback_spec",
    "load_snapback_solve_spec",
    "load_snapback_spec",
    "load_snapback_visual_spec",
    "resolve_snapback_workspace_relative_path",
    "resolve_workspace_root_for_released_snapback_spec",
    "resolve_workspace_root_for_snapback_solve_spec",
    "resolve_workspace_root_for_snapback_spec",
    "resolve_workspace_root_for_snapback_visual_spec",
]


def resolve_snapback_workspace_relative_path(raw_path: Path, *, workspace_root: Path, label: str) -> Path:
    return resolve_workspace_relative_path(raw_path, workspace_root=workspace_root, label=label)
