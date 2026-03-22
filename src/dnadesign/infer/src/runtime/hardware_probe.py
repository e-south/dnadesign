"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/src/runtime/hardware_probe.py

GPU inventory probing helpers for infer runtime capacity checks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GpuDeviceInfo:
    index: int
    name: str
    total_memory_gib: float
    compute_capability: str


@dataclass(frozen=True)
class GpuInventory:
    devices: tuple[GpuDeviceInfo, ...]

    @property
    def count(self) -> int:
        return len(self.devices)


def probe_gpu_inventory() -> GpuInventory:
    try:
        import torch
    except Exception:
        return GpuInventory(devices=())

    if not torch.cuda.is_available():
        return GpuInventory(devices=())

    devices: list[GpuDeviceInfo] = []
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        devices.append(
            GpuDeviceInfo(
                index=idx,
                name=str(props.name),
                total_memory_gib=float(props.total_memory) / float(1024**3),
                compute_capability=f"{props.major}.{props.minor}",
            )
        )
    return GpuInventory(devices=tuple(devices))


__all__ = ["GpuDeviceInfo", "GpuInventory", "probe_gpu_inventory"]
