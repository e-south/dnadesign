"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/status/service.py

Lazy status kind loading, input coercion, and provider dispatch.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from pathlib import Path

from .models import InputFieldSpec, StatusKindSpec
from .path_ref import PathBase, resolve_path_ref
from .provider_protocols import StatusProvider
from .registry_loader import list_status_kind_specs, load_status_kind_spec

_STATUS_STATES = frozenset({"ok", "attention", "missing"})


def build_status_inputs(
    *,
    spec: StatusKindSpec,
    raw_inputs: Mapping[str, object] | None,
    repo_root: Path | None,
    manifest_dir: Path | None = None,
    default_path_base: PathBase | None = None,
) -> dict[str, object]:
    provided_inputs = dict(raw_inputs or {})
    allowed_names = {field.name for field in spec.input_schema}
    unexpected = sorted(set(provided_inputs) - allowed_names)
    if unexpected:
        raise ValueError(f"status kind '{spec.status_kind}' does not accept inputs: {', '.join(unexpected)}")

    resolved_inputs: dict[str, object] = {}
    for field in spec.input_schema:
        if field.name not in provided_inputs:
            if field.default is not None:
                resolved_inputs[field.name] = field.default
                continue
            if field.required:
                raise ValueError(f"status kind '{spec.status_kind}' requires {field.cli_flag}")
            continue
        resolved_inputs[field.name] = _coerce_input_value(
            field,
            provided_inputs[field.name],
            repo_root=repo_root,
            manifest_dir=manifest_dir,
            default_path_base=default_path_base,
        )
    return resolved_inputs


def run_status_kind(
    status_kind: str,
    *,
    repo_root: Path | None,
    raw_inputs: Mapping[str, object] | None,
    manifest_dir: Path | None = None,
    default_path_base: PathBase | None = None,
) -> tuple[str, str, dict[str, object]]:
    spec = load_status_kind_spec(status_kind)
    resolved_inputs = build_status_inputs(
        spec=spec,
        raw_inputs=raw_inputs,
        repo_root=repo_root,
        manifest_dir=manifest_dir,
        default_path_base=default_path_base,
    )
    provider = _load_status_provider(spec.provider_ref)
    state, summary, evidence = provider(repo_root=repo_root, inputs=resolved_inputs)
    if state not in _STATUS_STATES:
        raise ValueError(f"invalid status state from {spec.provider_ref}: {state}")
    return state, summary, evidence


def _load_status_provider(provider_ref: str) -> StatusProvider:
    module_name, function_name = provider_ref.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    provider = getattr(module, function_name, None)
    if provider is None or not callable(provider):
        raise ValueError(f"status provider ref did not resolve to a callable: {provider_ref}")
    return provider


def _coerce_input_value(
    field: InputFieldSpec,
    raw_value: object,
    *,
    repo_root: Path | None,
    manifest_dir: Path | None,
    default_path_base: PathBase | None,
) -> object:
    if field.type == "path":
        return resolve_path_ref(
            raw_value,
            repo_root=repo_root,
            manifest_dir=manifest_dir,
            default_base=field.path_base or default_path_base,
            label=field.cli_flag,
        )
    if field.type == "int":
        try:
            return int(str(raw_value).strip())
        except ValueError as exc:
            raise ValueError(f"{field.cli_flag} must be an integer") from exc
    if field.type == "bool":
        if isinstance(raw_value, bool):
            return raw_value
        normalized = str(raw_value).strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"{field.cli_flag} must be one of: true, false, 1, 0, yes, no")
    text = str(raw_value).strip()
    if not text:
        raise ValueError(f"{field.cli_flag} must be non-empty")
    if field.type == "enum":
        if text not in field.choices:
            raise ValueError(f"{field.cli_flag} must be one of: {', '.join(field.choices)}")
    return text


__all__ = [
    "build_status_inputs",
    "list_status_kind_specs",
    "load_status_kind_spec",
    "run_status_kind",
]
