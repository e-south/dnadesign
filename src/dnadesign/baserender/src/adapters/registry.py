"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/adapters/registry.py

Central adapter registry for construction and source-column requirements.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from ..config import AdapterCfg
from ..config.adapter_contracts import adapter_contract
from ..core import Record, SchemaError
from .cruncher_best_window import CruncherBestWindowAdapter
from .densegen_tfbs import DensegenTfbsAdapter
from .generic_features import GenericFeaturesAdapter
from .sequence_windows_v1 import SequenceWindowsV1Adapter


class Adapter(Protocol):
    def apply(self, row: dict, *, row_index: int) -> Record: ...


AdapterFactory = Callable[[AdapterCfg, str], Adapter]


@dataclass(frozen=True)
class AdapterSpec:
    factory: AdapterFactory


def _build_densegen(cfg: AdapterCfg, alphabet: str) -> Adapter:
    return DensegenTfbsAdapter(columns=cfg.columns, policies=cfg.policies, alphabet=alphabet)


def _build_generic(cfg: AdapterCfg, alphabet: str) -> Adapter:
    return GenericFeaturesAdapter(columns=cfg.columns, policies=cfg.policies, alphabet=alphabet)


def _build_cruncher(cfg: AdapterCfg, alphabet: str) -> Adapter:
    return CruncherBestWindowAdapter.from_config(columns=cfg.columns, policies=cfg.policies, alphabet=alphabet)


def _build_sequence_windows(cfg: AdapterCfg, alphabet: str) -> Adapter:
    return SequenceWindowsV1Adapter(columns=cfg.columns, policies=cfg.policies, alphabet=alphabet)


ADAPTER_SPECS: dict[str, AdapterSpec] = {
    "densegen_tfbs": AdapterSpec(factory=_build_densegen),
    "generic_features": AdapterSpec(factory=_build_generic),
    "cruncher_best_window": AdapterSpec(factory=_build_cruncher),
    "sequence_windows_v1": AdapterSpec(factory=_build_sequence_windows),
}


def _get_spec(kind: str) -> AdapterSpec:
    spec = ADAPTER_SPECS.get(kind)
    if spec is None:
        raise SchemaError(f"Unsupported adapter kind: {kind}")
    return spec


def _append_unique(out: list[str], value: str) -> None:
    if value not in out:
        out.append(value)


def required_source_columns(adapter_cfg: AdapterCfg) -> list[str]:
    _get_spec(adapter_cfg.kind)
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
    spec = _get_spec(adapter_cfg.kind)
    return spec.factory(adapter_cfg, alphabet)
