"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/load.py

Load YIU specs and resolve workspace-relative paths.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from dnadesign.cruncher.yiu.models import YiuProcessSpec, YiuSpecDocument


def resolve_workspace_root_for_yiu_spec(spec_path: Path) -> Path:
    resolved = spec_path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"YIU spec not found: {resolved}")
    if not resolved.name.endswith(".yiu.yaml"):
        raise ValueError("--spec must point to a <workspace>/configs/yiu/<name>.yiu.yaml file.")
    for parent in resolved.parents:
        if parent.name == "configs":
            return parent.parent.resolve()
    raise ValueError("--spec must live under a workspace configs/ tree.")


def resolve_workspace_relative_path(raw_path: Path, *, workspace_root: Path, label: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    if any(part == ".." for part in path.parts):
        raise ValueError(f"{label} must not traverse outside the workspace: {raw_path}")
    return (workspace_root / path).resolve()


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in YIU spec {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"YIU spec {path} must be a YAML mapping.")
    return payload


def load_yiu_spec(path: str | Path) -> tuple[YiuProcessSpec, Path, Path]:
    spec_path = Path(path).expanduser().resolve()
    workspace_root = resolve_workspace_root_for_yiu_spec(spec_path)
    payload = _load_yaml_mapping(spec_path)
    if "yiu" not in payload:
        raise ValueError("YIU spec must define top-level key 'yiu'.")
    try:
        document = YiuSpecDocument.model_validate(payload)
    except Exception as exc:
        raise ValueError(f"YIU schema validation failed for {spec_path}: {exc}") from exc
    return document.yiu, spec_path, workspace_root
