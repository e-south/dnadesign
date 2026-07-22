"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/catalog/compiler_spec_io.py

Fail-fast Retron MSD compiler-spec file loading.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .strict_mapping_io import DuplicateMappingKeyError, load_unique_yaml


class MsdCompilerSpecError(ValueError):
    """Raised when a Retron MSD compiler spec is missing required intent."""


def load_compiler_spec_mapping(path: str | Path) -> dict[str, Any]:
    spec_path = Path(path).expanduser().resolve()
    if not spec_path.is_file():
        raise MsdCompilerSpecError(f"Retron MSD compiler spec not found: {spec_path}")
    try:
        if spec_path.suffix.lower() == ".json":
            payload = json.loads(spec_path.read_text(encoding="utf-8"), object_pairs_hook=_json_object_pairs)
        else:
            payload = load_unique_yaml(spec_path)
    except MsdCompilerSpecError:
        raise
    except DuplicateMappingKeyError as exc:
        raise MsdCompilerSpecError(f"Retron MSD compiler spec contains {exc}") from exc
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise MsdCompilerSpecError(f"Retron MSD compiler spec is invalid: {spec_path}") from exc
    if not isinstance(payload, dict):
        raise MsdCompilerSpecError(f"Retron MSD compiler spec must be a mapping: {spec_path}")
    return payload


def _json_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for key, value in pairs:
        if key in mapping:
            raise MsdCompilerSpecError(f"Retron MSD compiler spec contains duplicate mapping key: {key!r}.")
        mapping[key] = value
    return mapping


__all__ = ["MsdCompilerSpecError", "load_compiler_spec_mapping"]
