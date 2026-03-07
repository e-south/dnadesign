"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/src/runtime/capacity_planner.py

Model topology and GPU-capacity contract checks for infer execution.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from ..config import ModelConfig
from ..errors import ValidationError
from .hardware_probe import GpuDeviceInfo, GpuInventory, probe_gpu_inventory

_MODEL_PARAM_BILLIONS = {
    "evo2_1b_base": 1.0,
    "evo2_7b": 7.0,
    "evo2_20b": 20.0,
    "evo2_40b": 40.0,
}

_PRECISION_BYTES = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
}

_USABLE_MEMORY_FACTOR = 0.90
_MODEL_HEADROOM_FACTOR = 1.25


def estimate_required_gib(*, model_id: str, precision: str) -> float | None:
    params_b = _MODEL_PARAM_BILLIONS.get(str(model_id))
    bytes_per_param = _PRECISION_BYTES.get(str(precision))
    if params_b is None or bytes_per_param is None:
        return None
    weights_gib = params_b * 1e9 * bytes_per_param / float(1024**3)
    return weights_gib * _MODEL_HEADROOM_FACTOR


def _required_gpu_count(model: ModelConfig) -> int:
    if model.parallelism.strategy == "single_device":
        return 1
    return max(2, int(model.parallelism.min_gpus))


def _selected_devices(model: ModelConfig, inventory: GpuInventory) -> tuple[GpuDeviceInfo, ...]:
    if model.parallelism.gpu_ids is None:
        return inventory.devices

    invalid = [idx for idx in model.parallelism.gpu_ids if idx >= inventory.count]
    if invalid:
        raise ValidationError(
            "CAPACITY_FAIL "
            f"gpu_ids={model.parallelism.gpu_ids} "
            f"invalid_gpu_ids={invalid} "
            f"gpus_available={inventory.count}"
        )
    return tuple(inventory.devices[idx] for idx in model.parallelism.gpu_ids)


def validate_model_hardware_contract(
    *,
    model: ModelConfig,
    inventory: GpuInventory | None = None,
) -> None:
    device = str(model.device or "").strip()
    if model.parallelism.strategy == "multi_gpu_vortex" and not device.startswith("cuda"):
        raise ValidationError(
            "CAPACITY_FAIL "
            "parallelism.strategy=multi_gpu_vortex requires model.device to start with 'cuda'."
        )

    if not device.startswith("cuda"):
        return

    active_inventory = inventory or probe_gpu_inventory()
    required_gpus = _required_gpu_count(model)
    if active_inventory.count < required_gpus:
        raise ValidationError(
            "CAPACITY_FAIL "
            f"required_gpus={required_gpus} "
            f"gpus_available={active_inventory.count}"
        )

    selected = _selected_devices(model, active_inventory)
    if len(selected) < required_gpus:
        raise ValidationError(
            "CAPACITY_FAIL "
            f"gpu_ids={model.parallelism.gpu_ids} "
            f"required_gpus={required_gpus} "
            f"selected_gpus={len(selected)}"
        )
    selected = selected[:required_gpus]

    required_gib = estimate_required_gib(model_id=model.id, precision=model.precision)
    if required_gib is None:
        return

    usable_gib = sum(device_info.total_memory_gib * _USABLE_MEMORY_FACTOR for device_info in selected)
    if usable_gib < required_gib:
        raise ValidationError(
            "CAPACITY_FAIL "
            f"model_id={model.id} "
            f"precision={model.precision} "
            f"required_gib={required_gib:.1f} "
            f"usable_gib={usable_gib:.1f} "
            f"required_gpus={required_gpus} "
            f"gpus_available={active_inventory.count}"
        )


__all__ = [
    "GpuDeviceInfo",
    "GpuInventory",
    "estimate_required_gib",
    "probe_gpu_inventory",
    "validate_model_hardware_contract",
]
