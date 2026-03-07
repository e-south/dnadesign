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
}


def _load_model_config(config_path: Path):
    from pydantic import ValidationError as PydanticValidationError

    from .config import ModelConfig

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
        return ModelConfig(**model_payload)
    except (PydanticValidationError, ValueError) as exc:
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

    if memory_hint is None:
        required_gpus = model.parallelism.min_gpus if model.parallelism.strategy == "multi_gpu_vortex" else 1
        if resolved_declared_gpus < int(required_gpus):
            raise ValueError(
                "infer runbook resources do not satisfy model.parallelism contract: "
                f"required_gpus={required_gpus} resources.gpus={resolved_declared_gpus}"
            )
        if model.parallelism.gpu_ids is not None:
            invalid = [idx for idx in model.parallelism.gpu_ids if idx >= resolved_declared_gpus]
            if invalid:
                raise ValueError(
                    "infer runbook resources do not satisfy model.parallelism.gpu_ids contract: "
                    f"invalid_gpu_ids={invalid} resources.gpus={resolved_declared_gpus}"
                )
        return

    from .errors import ValidationError
    from .runtime.capacity_planner import GpuDeviceInfo, GpuInventory, validate_model_hardware_contract

    inventory = GpuInventory(
        devices=tuple(
            GpuDeviceInfo(
                index=index,
                name=f"declared_gpu_{index}",
                total_memory_gib=float(memory_hint),
                compute_capability=str(gpu_capability or ""),
            )
            for index in range(resolved_declared_gpus)
        )
    )
    try:
        validate_model_hardware_contract(model=model, inventory=inventory)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc


__all__ = ["validate_runbook_gpu_resources"]

