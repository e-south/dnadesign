"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/pwm_context_io.py

Shared file-loading and path-policy helpers for YIU PWM context sources.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from dnadesign.cruncher.yiu.errors import (
    YIU_PATH_INVALID,
    YIU_PWM_CONTEXT_INVALID,
    raise_yiu_error,
)


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise_yiu_error(YIU_PWM_CONTEXT_INVALID, f"invalid PWM context YAML at {path} ({exc})")
    if not isinstance(payload, dict):
        raise_yiu_error(YIU_PWM_CONTEXT_INVALID, f"PWM context file must be a YAML mapping: {path}")
    return payload


def resolve_workspace_file_path(raw_path: str, *, workspace_root: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    if any(part == ".." for part in path.parts):
        raise_yiu_error(YIU_PATH_INVALID, "optimization.pwm.source.path must not traverse outside the workspace")
    return (workspace_root / path).resolve()


__all__ = ["load_yaml_mapping", "resolve_workspace_file_path"]
