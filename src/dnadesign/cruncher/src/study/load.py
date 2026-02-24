"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/study/load.py

Load and validate Study spec files.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from dnadesign.cruncher.study.schema import parse_study_root
from dnadesign.cruncher.study.schema_models import StudySpec


def _resolve_base_config_root(spec_path: Path) -> Path:
    if spec_path.parent.name == "studies" and spec_path.parent.parent.name == "configs":
        return spec_path.parent.parent
    return spec_path.parent


def _format_schema_validation_error(exc: ValidationError) -> str:
    lines = ["Study schema validation failed:"]
    for error in exc.errors():
        loc = error.get("loc", ())
        loc_path = ".".join(str(item) for item in loc) if isinstance(loc, tuple | list) else str(loc)
        if not loc_path:
            loc_path = "<root>"
        message = str(error.get("msg", "invalid value")).strip()
        if message.startswith("Value error, "):
            message = message[len("Value error, ") :]
        lines.append(f"- {loc_path}: {message}")
    return "\n".join(lines)


def load_study_spec(path: Path) -> StudySpec:
    if not path.exists():
        raise FileNotFoundError(f"Study spec not found: {path}")
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or "study" not in raw:
        raise ValueError("Study schema v3 required (missing root key: study)")
    payload = raw.get("study")
    if not isinstance(payload, dict):
        raise ValueError("Study schema v3 required (study must be a mapping)")

    try:
        spec = parse_study_root(raw).study
    except ValidationError as exc:
        raise ValueError(_format_schema_validation_error(exc)) from exc
    base_config = spec.base_config
    base_root = _resolve_base_config_root(path)
    if base_config.is_absolute():
        resolved = base_config.resolve()
    else:
        resolved = (base_root / base_config).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Study base_config does not exist: {resolved}")

    normalized = spec.model_dump(mode="python")
    normalized["base_config"] = resolved
    try:
        return StudySpec.model_validate(normalized)
    except ValidationError as exc:
        raise ValueError(_format_schema_validation_error(exc)) from exc
