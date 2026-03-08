"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/tests/runtime/test_capacity_planner.py

Contract tests for infer GPU topology and capacity planning guards.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.infer.src.config import ModelConfig, ModelParallelismConfig
from dnadesign.infer.src.errors import ValidationError
from dnadesign.infer.src.runtime.capacity_planner import (
    GpuDeviceInfo,
    GpuInventory,
    validate_model_hardware_contract,
)


def _inventory(*, count: int, gib_per_gpu: float = 45.0, cc: str = "8.9") -> GpuInventory:
    return GpuInventory(
        devices=tuple(
            GpuDeviceInfo(
                index=i,
                name=f"gpu{i}",
                total_memory_gib=gib_per_gpu,
                compute_capability=cc,
            )
            for i in range(count)
        )
    )


def test_7b_single_device_passes_on_single_45g_gpu() -> None:
    model = ModelConfig(
        id="evo2_7b",
        device="cuda:0",
        precision="bf16",
        alphabet="dna",
        parallelism=ModelParallelismConfig(strategy="single_device"),
    )

    validate_model_hardware_contract(model=model, inventory=_inventory(count=1))


def test_40b_single_device_fails_on_single_45g_gpu() -> None:
    model = ModelConfig(
        id="evo2_40b",
        device="cuda:0",
        precision="bf16",
        alphabet="dna",
        parallelism=ModelParallelismConfig(strategy="single_device"),
    )

    with pytest.raises(ValidationError, match="CAPACITY_FAIL"):
        validate_model_hardware_contract(model=model, inventory=_inventory(count=1))


def test_20b_fails_on_non_hopper_gpu_even_when_memory_is_sufficient() -> None:
    model = ModelConfig(
        id="evo2_20b",
        device="cuda:0",
        precision="bf16",
        alphabet="dna",
        parallelism=ModelParallelismConfig(strategy="single_device"),
    )

    with pytest.raises(ValidationError, match="requires Hopper"):
        validate_model_hardware_contract(model=model, inventory=_inventory(count=1, gib_per_gpu=80.0, cc="8.9"))


def test_40b_multi_gpu_vortex_fails_on_non_hopper_gpus_even_when_aggregate_capacity_is_sufficient() -> None:
    model = ModelConfig(
        id="evo2_40b",
        device="cuda:0",
        precision="bf16",
        alphabet="dna",
        parallelism=ModelParallelismConfig(strategy="multi_gpu_vortex", min_gpus=3),
    )

    with pytest.raises(ValidationError, match="requires Hopper"):
        validate_model_hardware_contract(model=model, inventory=_inventory(count=3, gib_per_gpu=45.0, cc="8.9"))


def test_multi_gpu_vortex_requires_at_least_two_gpus() -> None:
    model = ModelConfig(
        id="evo2_40b",
        device="cuda:0",
        precision="bf16",
        alphabet="dna",
        parallelism=ModelParallelismConfig(strategy="multi_gpu_vortex", min_gpus=2),
    )

    with pytest.raises(ValidationError, match="required_gpus=2"):
        validate_model_hardware_contract(model=model, inventory=_inventory(count=1))


def test_40b_multi_gpu_vortex_passes_when_aggregate_capacity_is_sufficient() -> None:
    model = ModelConfig(
        id="evo2_40b",
        device="cuda:0",
        precision="bf16",
        alphabet="dna",
        parallelism=ModelParallelismConfig(strategy="multi_gpu_vortex", min_gpus=3),
    )

    validate_model_hardware_contract(model=model, inventory=_inventory(count=3, cc="9.0"))


def test_multi_gpu_vortex_fails_when_gpu_ids_do_not_cover_required_gpus() -> None:
    model = ModelConfig(
        id="evo2_40b",
        device="cuda:0",
        precision="bf16",
        alphabet="dna",
        parallelism=ModelParallelismConfig(
            strategy="multi_gpu_vortex",
            min_gpus=2,
            gpu_ids=[0, 5],
        ),
    )

    with pytest.raises(ValidationError, match="invalid_gpu_ids"):
        validate_model_hardware_contract(model=model, inventory=_inventory(count=4))


def test_single_device_fails_when_device_index_is_out_of_range() -> None:
    model = ModelConfig(
        id="evo2_7b",
        device="cuda:3",
        precision="bf16",
        alphabet="dna",
        parallelism=ModelParallelismConfig(strategy="single_device"),
    )

    with pytest.raises(ValidationError, match="device_index=3"):
        validate_model_hardware_contract(model=model, inventory=_inventory(count=2))
