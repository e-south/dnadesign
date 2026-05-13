"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/scar_nick/load.py

Load scar-nick specs and resolve workspace roots.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from dnadesign.cruncher.scar_nick.errors import ScarNickSpecError
from dnadesign.cruncher.scar_nick.models import ScarNickSpecDocument


def resolve_workspace_root_for_scar_nick_spec(spec_path: Path) -> Path:
    resolved = spec_path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Scar-nick spec not found: {resolved}")
    if not resolved.name.endswith(".scar_nick.yaml"):
        raise ScarNickSpecError("--spec must point to a <workspace>/configs/scar_nick/<name>.scar_nick.yaml file.")
    if len(resolved.parents) < 3 or resolved.parent.name != "scar_nick" or resolved.parent.parent.name != "configs":
        raise ScarNickSpecError("--spec must point to a <workspace>/configs/scar_nick/<name>.scar_nick.yaml file.")
    return resolved.parent.parent.parent.resolve()


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ScarNickSpecError(f"Invalid YAML in scar-nick spec {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ScarNickSpecError(f"Scar-nick spec {path} must be a YAML mapping.")
    return payload


def load_scar_nick_spec(path: str | Path) -> tuple[ScarNickSpecDocument, Path, Path]:
    spec_path = Path(path).expanduser().resolve()
    workspace_root = resolve_workspace_root_for_scar_nick_spec(spec_path)
    payload = _load_yaml_mapping(spec_path)
    try:
        document = ScarNickSpecDocument.model_validate(payload)
    except Exception as exc:
        raise ScarNickSpecError(f"Scar-nick schema validation failed for {spec_path}: {exc}") from exc
    return document, spec_path, workspace_root


__all__ = ["load_scar_nick_spec", "resolve_workspace_root_for_scar_nick_spec"]
