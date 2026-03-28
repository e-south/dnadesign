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

from dnadesign.cruncher.yiu.models import (
    YiuProcessSpec,
    YiuProcessSpecV2,
    YiuSolveSpec,
    YiuSolveSpecDocument,
    YiuSpecDocument,
    YiuSpecDocumentV2,
    deprecated_yiu_protocol_template_alias,
)


def _resolve_workspace_root_for_suffix(spec_path: Path, *, suffix: str, help_message: str) -> Path:
    resolved = spec_path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"YIU spec not found: {resolved}")
    if not resolved.name.endswith(suffix):
        raise ValueError(help_message)
    if len(resolved.parents) < 3 or resolved.parent.name != "yiu" or resolved.parent.parent.name != "configs":
        raise ValueError(help_message)
    return resolved.parent.parent.parent.resolve()


def resolve_workspace_root_for_yiu_spec(spec_path: Path) -> Path:
    return _resolve_workspace_root_for_suffix(
        spec_path,
        suffix=".yiu.yaml",
        help_message="--spec must point to a <workspace>/configs/yiu/<name>.yiu.yaml file.",
    )


def resolve_workspace_root_for_yiu_solve_spec(spec_path: Path) -> Path:
    return _resolve_workspace_root_for_suffix(
        spec_path,
        suffix=".yiu.solve.yaml",
        help_message="--spec must point to a <workspace>/configs/yiu/<name>.yiu.solve.yaml file.",
    )


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


def load_yiu_spec(path: str | Path) -> tuple[YiuProcessSpec | YiuProcessSpecV2, Path, Path]:
    spec_path = Path(path).expanduser().resolve()
    workspace_root = resolve_workspace_root_for_yiu_spec(spec_path)
    payload = _load_yaml_mapping(spec_path)
    if "yiu" not in payload:
        raise ValueError("YIU spec must define top-level key 'yiu'.")
    raw_root = payload["yiu"]
    if not isinstance(raw_root, dict):
        raise ValueError("YIU spec root 'yiu' must be a mapping.")
    raw_schema_version = raw_root.get("schema_version", 1)
    try:
        schema_version = int(raw_schema_version)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"YIU schema validation failed for {spec_path}: invalid schema_version {raw_schema_version!r}"
        ) from exc
    try:
        if schema_version == 1:
            document = YiuSpecDocument.model_validate(payload)
        elif schema_version == 2:
            document = YiuSpecDocumentV2.model_validate(payload)
        else:
            raise ValueError(f"Unsupported YIU schema_version: {schema_version}")
    except Exception as exc:
        raise ValueError(f"YIU schema validation failed for {spec_path}: {exc}") from exc
    spec = document.yiu
    if isinstance(spec, YiuProcessSpecV2):
        alias_used = deprecated_yiu_protocol_template_alias(str(raw_root.get("protocol_template") or ""))
        if alias_used is not None:
            spec = spec.model_copy(
                update={
                    "template_alias_used": alias_used,
                    "template_alias_status": "deprecated_alias",
                }
            )
    return spec, spec_path, workspace_root


def load_yiu_solve_spec(path: str | Path) -> tuple[YiuSolveSpec, Path, Path]:
    spec_path = Path(path).expanduser().resolve()
    workspace_root = resolve_workspace_root_for_yiu_solve_spec(spec_path)
    payload = _load_yaml_mapping(spec_path)
    if "yiu_solve" not in payload:
        raise ValueError("YIU solve spec must define top-level key 'yiu_solve'.")
    raw_root = payload["yiu_solve"]
    if not isinstance(raw_root, dict):
        raise ValueError("YIU solve spec root 'yiu_solve' must be a mapping.")
    try:
        document = YiuSolveSpecDocument.model_validate(payload)
    except Exception as exc:
        raise ValueError(f"YIU solve schema validation failed for {spec_path}: {exc}") from exc
    return document.yiu_solve, spec_path, workspace_root


def resolve_base_spec_path_for_yiu_solve_spec(solve_spec: YiuSolveSpec, *, workspace_root: Path) -> Path:
    base_spec_path = resolve_workspace_relative_path(
        solve_spec.base_spec,
        workspace_root=workspace_root,
        label="yiu_solve.base_spec",
    )
    resolve_workspace_root_for_yiu_spec(base_spec_path)
    return base_spec_path
