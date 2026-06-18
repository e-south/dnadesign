"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/operations/status/probes/runtime_dependencies.py

Runtime dependency probes for the stress_ethanol_cipro_growth status service.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from dnadesign.ops.status import (
    load_yaml_mapping,
    resolve_named_path_mapping,
    string_list_or_empty,
    string_or_none,
)

from ..infer_runtime import StressEthanolCiproGrowthInferRuntimeDependencies


def build_stress_ethanol_cipro_growth_infer_runtime_dependencies() -> StressEthanolCiproGrowthInferRuntimeDependencies:
    from dnadesign.infer import resolve_infer_runtime_lane_contracts

    return StressEthanolCiproGrowthInferRuntimeDependencies(
        resolve_named_path_mapping=resolve_named_path_mapping,
        resolve_infer_runtime_lane_contracts=resolve_infer_runtime_lane_contracts,
        derive_infer_notify_profile_paths=derive_infer_notify_profile_paths,
        load_infer_model_summary=load_infer_model_summary,
        string_or_none=string_or_none,
        string_list_or_empty=string_list_or_empty,
    )


def inspect_local_infer_gpu_inventory() -> dict[str, object]:
    try:
        from dnadesign.infer import inspect_local_gpu_inventory

        payload = inspect_local_gpu_inventory()
    except Exception as exc:
        return {"count": 0, "devices": [], "probe_error": str(exc)}
    if not isinstance(payload, dict):
        return {"count": 0, "devices": [], "probe_error": "infer.inspect_local_gpu_inventory returned invalid data"}
    devices = payload.get("devices")
    resolved_devices = list(devices) if isinstance(devices, list) else []
    return {
        "count": int(payload.get("count") or len(resolved_devices)),
        "devices": resolved_devices,
        "probe_error": string_or_none(payload.get("probe_error")),
    }


def derive_infer_notify_profile_paths(
    infer_config_paths: Mapping[str, Path],
) -> tuple[dict[str, Path], dict[str, str]]:
    if not infer_config_paths:
        return {}, {}
    from dnadesign.infer.contracts import resolve_infer_notify_profile_path

    derived_paths: dict[str, Path] = {}
    derivation_errors: dict[str, str] = {}
    for config_label, config_path in infer_config_paths.items():
        try:
            derived_paths[config_label] = resolve_infer_notify_profile_path(config_path)
        except Exception as exc:
            derivation_errors[config_label] = str(exc)
    return derived_paths, derivation_errors


def load_infer_model_summary(config_path: Path) -> dict[str, object]:
    payload = load_yaml_mapping(config_path, label="infer config")
    model_payload = payload.get("model") or {}
    if not isinstance(model_payload, dict):
        raise ValueError(f"infer config must define a model mapping: {config_path}")
    return {
        "model_id": string_or_none(model_payload.get("id")),
        "device": string_or_none(model_payload.get("device")) or "unknown",
    }


def phase_matches_infer_model_family(*, phase_id: str, model_family: str | None) -> bool:
    from dnadesign.infer import infer_model_family_suffix

    suffix = infer_model_family_suffix(model_family)
    return suffix is not None and suffix in phase_id


__all__ = [
    "build_stress_ethanol_cipro_growth_infer_runtime_dependencies",
    "inspect_local_infer_gpu_inventory",
    "phase_matches_infer_model_family",
]
