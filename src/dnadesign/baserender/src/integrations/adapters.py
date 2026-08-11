"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/integrations/adapters.py

Adapter construction and policy checks backed by integration descriptors.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal, Protocol

from ..core import ContractError, Record, SchemaError, reject_unknown_keys
from .contracts import AdapterDescriptor, AdapterPathResolver
from .registry import registered_adapter, registered_adapters


class Adapter(Protocol):
    def apply(self, row: dict, *, row_index: int) -> Record: ...


def adapter_kinds() -> set[str]:
    return {descriptor.kind for descriptor in registered_adapters()}


def adapter_descriptors() -> tuple[AdapterDescriptor, ...]:
    return registered_adapters()


def adapter_descriptor(kind: str) -> AdapterDescriptor:
    descriptor = registered_adapter(kind)
    if descriptor is None:
        raise SchemaError(f"Unsupported adapter kind: {kind}")
    return descriptor


adapter_contract = adapter_descriptor


def declared_adapter_path_values(kind: Any, columns: Mapping[str, Any]) -> tuple[Any, ...]:
    descriptor = registered_adapter(str(kind))
    if descriptor is None:
        return ()
    return tuple(columns[key] for key in descriptor.resolved_path_columns if columns.get(key) is not None)


def normalize_adapter_config(
    *,
    kind: Any,
    columns: Mapping[str, Any],
    policies: Mapping[str, Any],
    alphabet: str | None = None,
    resolve_path: AdapterPathResolver | None = None,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    try:
        parsed_kind = str(kind).strip()
        contract = adapter_descriptor(parsed_kind)

        reject_unknown_keys(columns, set(contract.allowed_config_columns), "input.adapter.columns")
        missing = sorted(set(contract.required_config_columns) - set(columns))
        if missing:
            raise SchemaError(f"input.adapter.columns missing required keys for {parsed_kind}: {missing}")

        parsed_columns = dict(columns)
        if resolve_path is not None:
            for key in contract.resolved_path_columns:
                if parsed_columns.get(key) is not None:
                    parsed_columns[key] = resolve_path(key, parsed_columns[key])

        reject_unknown_keys(policies, set(contract.allowed_policy_keys), "input.adapter.policies")
        parsed_policies = contract.normalize_policies(policies, "input.adapter.policies")

        if alphabet is not None and alphabet not in contract.supported_alphabets:
            allowed = ", ".join(sorted(contract.supported_alphabets))
            raise SchemaError(
                f"input.adapter.kind {parsed_kind!r} is not compatible with input.alphabet {alphabet!r}; "
                f"supported input.alphabet values: {allowed}"
            )
        return parsed_kind, parsed_columns, parsed_policies
    except ContractError as exc:
        raise SchemaError(str(exc)) from exc


def required_source_columns(adapter_cfg) -> list[str]:
    contract = adapter_descriptor(adapter_cfg.kind)
    columns = adapter_cfg.columns
    out: list[str] = []
    for key in contract.required_source_columns:
        if columns.get(key) is None:
            raise SchemaError(f"missing required adapter column key '{key}' for adapter '{adapter_cfg.kind}'")
        value = str(columns[key])
        if value not in out:
            out.append(value)
    for key in contract.optional_source_columns:
        if columns.get(key) is not None:
            value = str(columns[key])
            if value not in out:
                out.append(value)
    return out


def build_adapter(adapter_cfg, *, alphabet: str) -> Adapter:
    return adapter_descriptor(adapter_cfg.kind).factory(adapter_cfg, alphabet)


def finalize_adapter(adapter: Adapter) -> None:
    finalizer = getattr(adapter, "finalize", None)
    if finalizer is not None:
        finalizer()


def validate_adapter_output_policy(
    adapter_kind: str,
    *,
    output_kind: Literal["images", "video"],
    image_output_mode: Literal["directory", "single_file"] | None = None,
) -> None:
    descriptor = adapter_descriptor(adapter_kind)
    if output_kind not in descriptor.output_kinds:
        allowed = ", ".join(descriptor.output_kinds)
        raise SchemaError(f"adapter {adapter_kind!r} only supports output kinds: {allowed}")
    if output_kind != "images":
        return
    if image_output_mode is None:
        raise SchemaError("image_output_mode is required for images output")
    if image_output_mode not in descriptor.image_output_modes:
        if descriptor.image_output_modes == ("directory",):
            raise SchemaError(
                f"adapter {adapter_kind!r} requires a directory for images output; single-file images are not supported"
            )
        allowed = ", ".join(descriptor.image_output_modes)
        raise SchemaError(
            f"adapter {adapter_kind!r} does not support images output mode {image_output_mode!r}; "
            f"supported modes: {allowed}"
        )


def _record_adapter_kind(record: Record, *, record_index: int) -> str | None:
    if not isinstance(record.meta, Mapping):
        raise SchemaError(f"records[{record_index}].meta must be a mapping/dict")
    raw_kind = record.meta.get("adapter")
    if raw_kind is None:
        return None
    if not isinstance(raw_kind, str) or not raw_kind.strip():
        raise SchemaError(f"records[{record_index}].meta.adapter must be a non-empty string")
    kind = raw_kind.strip()
    adapter_descriptor(kind)
    return kind


def adapter_grid_record_limit(records: Iterable[Record]) -> int | None:
    limit: int | None = None
    for record_index, record in enumerate(records):
        kind = _record_adapter_kind(record, record_index=record_index)
        if kind is None:
            continue
        adapter_limit = adapter_descriptor(kind).max_grid_records
        if adapter_limit is not None:
            limit = adapter_limit if limit is None else min(limit, adapter_limit)
    return limit


def validate_records_output_policy(
    records: Iterable[Record],
    *,
    output_kind: Literal["images", "video"],
    image_output_mode: Literal["directory", "single_file"] | None = None,
) -> None:
    for record_index, record in enumerate(records):
        kind = _record_adapter_kind(record, record_index=record_index)
        if kind is not None:
            validate_adapter_output_policy(
                kind,
                output_kind=output_kind,
                image_output_mode=image_output_mode,
            )


def validate_record_renderer_compatibility(record: Record, *, renderer_name: str) -> None:
    kind = _record_adapter_kind(record, record_index=0)
    if kind is None:
        return
    supported_renderers = adapter_descriptor(kind).supported_renderers
    if renderer_name not in supported_renderers:
        allowed = ", ".join(sorted(supported_renderers))
        raise SchemaError(
            f"record.meta.adapter {kind!r} is not compatible with renderer {renderer_name!r}; "
            f"supported renderer values: {allowed}"
        )


__all__ = [
    "Adapter",
    "adapter_contract",
    "adapter_descriptor",
    "adapter_descriptors",
    "adapter_grid_record_limit",
    "adapter_kinds",
    "build_adapter",
    "declared_adapter_path_values",
    "finalize_adapter",
    "normalize_adapter_config",
    "required_source_columns",
    "validate_adapter_output_policy",
    "validate_record_renderer_compatibility",
    "validate_records_output_policy",
]
