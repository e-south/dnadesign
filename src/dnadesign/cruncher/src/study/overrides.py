"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/study/overrides.py

Apply and flatten Study dot-path factors against Cruncher config objects.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import math
from typing import Any

from pydantic import ValidationError

from dnadesign.cruncher.config.schema_v3 import CruncherConfig


def _resolve_parent(payload: dict[str, Any], path_parts: list[str]) -> dict[str, Any]:
    cursor: Any = payload
    for idx, part in enumerate(path_parts):
        if not isinstance(cursor, dict):
            prefix = ".".join(path_parts[:idx])
            raise ValueError(f"Unknown override path '{'.'.join(path_parts)}' (segment '{prefix}' is not a mapping)")
        if part not in cursor:
            raise ValueError(f"Unknown override path '{'.'.join(path_parts)}'")
        cursor = cursor[part]
    if not isinstance(cursor, dict):
        raise ValueError(f"Unknown override path '{'.'.join(path_parts)}' (leaf parent is not a mapping)")
    return cursor


def apply_dotpath_overrides(cfg: CruncherConfig, overrides: dict[str, Any]) -> CruncherConfig:
    payload = cfg.model_dump(mode="python")
    for raw_path, value in overrides.items():
        path = str(raw_path).strip()
        if not path:
            raise ValueError("Override paths must be non-empty strings.")
        parts = path.split(".")
        if any(not part for part in parts):
            raise ValueError(f"Override path is invalid: {raw_path!r}")
        parent = _resolve_parent(payload, parts[:-1]) if len(parts) > 1 else payload
        leaf = parts[-1]
        if leaf not in parent:
            raise ValueError(f"Unknown override path '{path}'")
        parent[leaf] = value

    try:
        return CruncherConfig.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"Invalid study override payload: {exc}") from exc


def _canonical_json(value: Any, *, context: str) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Study override value for {context} must be JSON-serializable and finite.") from exc


def _column_value(value: Any, *, path: str) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Study override value for '{path}' must be finite.")
        return float(value)
    return _canonical_json(value, context=f"'{path}'")


def extract_factor_columns(factors: dict[str, Any]) -> dict[str, Any]:
    columns: dict[str, Any] = {}
    for path in sorted(factors.keys()):
        column = "param__" + "__".join(path.split("."))
        columns[column] = _column_value(factors[path], path=path)
    ordered_factors = {path: factors[path] for path in sorted(factors.keys())}
    columns["params_json"] = _canonical_json(ordered_factors, context="params_json")
    return columns
