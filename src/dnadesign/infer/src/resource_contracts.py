"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/src/resource_contracts.py

Public infer resource-contract checks used by orchestration preflight workflows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

_GPU_CAPABILITY_MEMORY_HINT_GIB: dict[str, float] = {
    "8.9": 45.0,
    "9.0": 80.0,
}


def _load_model_config(config_path: Path):
    from pydantic import ValidationError as PydanticValidationError

    from .bootstrap import initialize_registry
    from .config import ModelConfig
    from .errors import ConfigError
    from .registry import get_adapter_cls

    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"infer config is not readable: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"infer config is not valid yaml: {config_path}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"infer config root must be a mapping: {config_path}")
    model_payload = payload.get("model")
    if not isinstance(model_payload, dict):
        raise ValueError(f"infer config must include a model block: {config_path}")
    try:
        model = ModelConfig(**model_payload)
        initialize_registry()
        get_adapter_cls(model.id)
        return model
    except (PydanticValidationError, ValueError, ConfigError) as exc:
        raise ValueError(f"infer model contract invalid in config {config_path}: {exc}") from exc


def _gpu_memory_hint(*, gpu_capability: str | None, gpu_memory_gib: float | None) -> float | None:
    if gpu_memory_gib is not None:
        return float(gpu_memory_gib)
    if gpu_capability is None:
        return None
    return _GPU_CAPABILITY_MEMORY_HINT_GIB.get(str(gpu_capability).strip())


def validate_runbook_gpu_resources(
    *,
    config_path: Path,
    declared_gpus: int,
    gpu_capability: str | None,
    gpu_memory_gib: float | None,
) -> None:
    if int(declared_gpus) <= 0:
        raise ValueError("declared_gpus must be >= 1")

    model = _load_model_config(Path(config_path))
    resolved_declared_gpus = int(declared_gpus)
    memory_hint = _gpu_memory_hint(gpu_capability=gpu_capability, gpu_memory_gib=gpu_memory_gib)

    from .errors import ValidationError
    from .runtime.adapter_runtime import validate_adapter_runtime_contract
    from .runtime.capacity_planner import (
        GpuDeviceInfo,
        GpuInventory,
        validate_model_gpu_topology_contract,
        validate_model_hardware_contract,
    )

    inventory = GpuInventory(
        devices=tuple(
            GpuDeviceInfo(
                index=index,
                name=f"declared_gpu_{index}",
                total_memory_gib=float(memory_hint or 0.0),
                compute_capability=str(gpu_capability or ""),
            )
            for index in range(resolved_declared_gpus)
        )
    )
    try:
        if memory_hint is None:
            validate_model_gpu_topology_contract(model=model, inventory=inventory)
        validate_model_hardware_contract(
            model=model,
            inventory=inventory,
            enforce_memory_capacity=memory_hint is not None,
        )
        validate_adapter_runtime_contract(model=model)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc


__all__ = ["validate_runbook_gpu_resources"]
