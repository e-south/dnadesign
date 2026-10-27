"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/yiu/load.py

Load payload-centric YIU v4 specs and resolve workspace-relative paths.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from dnadesign.cruncher.yiu.errors import (
    YIU_CONTRACT_UNKNOWN,
    YIU_PATH_INVALID,
    YIU_SCHEMA_VERSION_UNSUPPORTED,
    raise_yiu_error,
)
from dnadesign.cruncher.yiu.spec_models import YiuPayloadRenderingSpec


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


def resolve_workspace_relative_path(raw_path: Path, *, workspace_root: Path, label: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    if any(part == ".." for part in path.parts):
        raise ValueError(f"{YIU_PATH_INVALID}: {label} must not traverse outside the workspace")
    return (workspace_root / path).resolve()


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in YIU spec {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"YIU spec {path} must be a YAML mapping.")
    return payload


def load_yiu_spec(path: str | Path) -> tuple[YiuPayloadRenderingSpec, Path, Path]:
    spec_path = Path(path).expanduser().resolve()
    workspace_root = resolve_workspace_root_for_yiu_spec(spec_path)
    payload = _load_yaml_mapping(spec_path)
    raw_root = payload.get("yiu")
    if not isinstance(raw_root, dict):
        raise_yiu_error(YIU_CONTRACT_UNKNOWN, "YIU spec must define a top-level mapping key 'yiu'.")
    contract = raw_root.get("contract")
    if contract != "split_yiu_payload_rendering_v4":
        raise_yiu_error(
            YIU_CONTRACT_UNKNOWN,
            "YIU now writes native v4 specs only. Set yiu.contract=split_yiu_payload_rendering_v4 "
            "and migrate legacy v3 fields into input/optimization/output.",
        )
    raw_schema_version = raw_root.get("schema_version")
    try:
        schema_version = int(raw_schema_version)
    except (TypeError, ValueError):
        raise_yiu_error(YIU_SCHEMA_VERSION_UNSUPPORTED, f"invalid schema_version {raw_schema_version!r}")
    if schema_version != 1:
        raise_yiu_error(
            YIU_SCHEMA_VERSION_UNSUPPORTED,
            "split_yiu_payload_rendering_v4 only supports schema_version=1.",
        )
    try:
        document = YiuPayloadRenderingSpec.model_validate(payload)
    except Exception as exc:
        raise ValueError(f"YIU schema validation failed for {spec_path}: {exc}") from exc
    return document, spec_path, workspace_root
