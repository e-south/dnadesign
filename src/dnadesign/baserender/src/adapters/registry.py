"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/adapters/registry.py

Central adapter registry for construction and source-column requirements.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Callable, Protocol

from ..config import AdapterCfg
from ..config.adapter_contracts import AdapterDescriptor, adapter_contract, adapter_descriptor, adapter_descriptors
from ..core import Record, SchemaError


class Adapter(Protocol):
    def apply(self, row: dict, *, row_index: int) -> Record: ...


AdapterFactory = Callable[[AdapterCfg, str], Adapter]


def _append_unique(out: list[str], value: str) -> None:
    if value not in out:
        out.append(value)


def required_source_columns(adapter_cfg: AdapterCfg) -> list[str]:
    contract = adapter_contract(adapter_cfg.kind)
    cols = adapter_cfg.columns

    out: list[str] = []
    for key in contract.required_source_columns:
        if key not in cols or cols[key] is None:
            raise SchemaError(f"missing required adapter column key '{key}' for adapter '{adapter_cfg.kind}'")
        _append_unique(out, str(cols[key]))

    for key in contract.optional_source_columns:
        if key in cols and cols[key] is not None:
            _append_unique(out, str(cols[key]))

    return out


def build_adapter(adapter_cfg: AdapterCfg, *, alphabet: str) -> Adapter:
    descriptor = adapter_descriptor(adapter_cfg.kind)
    return descriptor.factory(adapter_cfg, alphabet)


def list_adapter_descriptors() -> tuple[AdapterDescriptor, ...]:
    return adapter_descriptors()


def get_adapter_descriptor(kind: str) -> AdapterDescriptor:
    return adapter_descriptor(kind)
