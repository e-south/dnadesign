"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/config/adapter_contracts.py

Canonical adapter contracts used by job parsing and adapter registry wiring.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

from ..core import SchemaError, ensure, require_one_of

PolicyNormalizer = Callable[[Mapping[str, Any], str], dict[str, Any]]


def _normalize_policies_passthrough(policies: Mapping[str, Any], _ctx: str) -> dict[str, Any]:
    return dict(policies)


def _normalize_densegen_policies(policies: Mapping[str, Any], ctx: str) -> dict[str, Any]:
    parsed = dict(policies)
    if "ambiguous" in parsed:
        require_one_of(
            str(parsed["ambiguous"]).lower(),
            {"error", "first", "last", "drop"},
            f"{ctx}.ambiguous",
        )
    if "offset_mode" in parsed:
        require_one_of(
            str(parsed["offset_mode"]).lower(),
            {"auto", "zero_based", "one_based"},
            f"{ctx}.offset_mode",
        )
    if "on_missing_kmer" in parsed:
        require_one_of(
            str(parsed["on_missing_kmer"]).lower(),
            {"error", "skip_entry"},
            f"{ctx}.on_missing_kmer",
        )
    if "on_invalid_row" in parsed:
        require_one_of(
            str(parsed["on_invalid_row"]).lower(),
            {"skip", "error"},
            f"{ctx}.on_invalid_row",
        )
    if "min_per_record" in parsed:
        min_per_record = int(parsed["min_per_record"])
        ensure(min_per_record >= 0, f"{ctx}.min_per_record must be >= 0", SchemaError)
        parsed["min_per_record"] = min_per_record
    if "require_non_null_cols" in parsed:
        cols = parsed["require_non_null_cols"]
        if not isinstance(cols, (list, tuple)):
            raise SchemaError(f"{ctx}.require_non_null_cols must be a list")
        parsed["require_non_null_cols"] = [str(c) for c in cols]
    for key in ("zero_as_unspecified", "require_non_empty"):
        if key in parsed:
            val = parsed[key]
            if not isinstance(val, bool):
                raise SchemaError(f"{ctx}.{key} must be bool")
            parsed[key] = val
    return parsed


def _normalize_cruncher_policies(policies: Mapping[str, Any], ctx: str) -> dict[str, Any]:
    parsed = dict(policies)
    if "on_missing_hit" in parsed:
        require_one_of(
            str(parsed["on_missing_hit"]).lower(),
            {"error", "skip"},
            f"{ctx}.on_missing_hit",
        )
    if "on_missing_pwm" in parsed:
        require_one_of(
            str(parsed["on_missing_pwm"]).lower(),
            {"error", "skip_effect"},
            f"{ctx}.on_missing_pwm",
        )
    return parsed


@dataclass(frozen=True)
class AdapterContract:
    allowed_config_columns: tuple[str, ...]
    required_config_columns: tuple[str, ...]
    required_source_columns: tuple[str, ...]
    optional_source_columns: tuple[str, ...] = ()
    allowed_policy_keys: tuple[str, ...] = ()
    resolved_path_columns: tuple[str, ...] = ()
    normalize_policies: PolicyNormalizer = _normalize_policies_passthrough


_DENSEGEN_POLICY_KEYS = (
    "ambiguous",
    "offset_mode",
    "zero_as_unspecified",
    "on_missing_kmer",
    "require_non_empty",
    "min_per_record",
    "require_non_null_cols",
    "on_invalid_row",
)

_CRUNCHER_POLICY_KEYS = ("on_missing_hit", "on_missing_pwm")


ADAPTER_CONTRACTS: dict[str, AdapterContract] = {
    "densegen_tfbs": AdapterContract(
        allowed_config_columns=("sequence", "annotations", "id", "overlay_text"),
        required_config_columns=("sequence", "annotations"),
        required_source_columns=("sequence", "annotations"),
        optional_source_columns=("id", "overlay_text"),
        allowed_policy_keys=_DENSEGEN_POLICY_KEYS,
        normalize_policies=_normalize_densegen_policies,
    ),
    "generic_features": AdapterContract(
        allowed_config_columns=("sequence", "features", "effects", "display", "id"),
        required_config_columns=("sequence", "features"),
        required_source_columns=("sequence", "features"),
        optional_source_columns=("effects", "display", "id"),
        normalize_policies=_normalize_policies_passthrough,
    ),
    "cruncher_best_window": AdapterContract(
        allowed_config_columns=(
            "sequence",
            "id",
            "hits_path",
            "hits_elite_id",
            "hits_tf",
            "hits_start",
            "hits_strand",
            "hits_window_seq",
            "hits_core_seq",
            "config_path",
        ),
        required_config_columns=("sequence", "id", "hits_path", "config_path"),
        required_source_columns=("sequence", "id"),
        allowed_policy_keys=_CRUNCHER_POLICY_KEYS,
        resolved_path_columns=("hits_path", "config_path"),
        normalize_policies=_normalize_cruncher_policies,
    ),
    "sequence_windows_v1": AdapterContract(
        allowed_config_columns=("id", "sequence", "regulator_windows", "motifs", "display"),
        required_config_columns=("sequence", "regulator_windows"),
        required_source_columns=("sequence", "regulator_windows"),
        optional_source_columns=("id", "motifs", "display"),
        normalize_policies=_normalize_policies_passthrough,
    ),
}


def adapter_kinds() -> set[str]:
    return set(ADAPTER_CONTRACTS.keys())


def adapter_contract(kind: str) -> AdapterContract:
    contract = ADAPTER_CONTRACTS.get(kind)
    if contract is None:
        raise SchemaError(f"Unsupported adapter kind: {kind}")
    return contract
