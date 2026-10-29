"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/src/runtime/hardware_probe.py

GPU inventory probing helpers for infer runtime capacity checks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


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


def _cuda_visible_devices_disables_gpu() -> bool:
    raw = str(os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip().lower()
    return raw in {"-1", "none", "void"}


def _gpu_host_candidate_exists() -> bool:
    if _cuda_visible_devices_disables_gpu():
        return False
    if Path("/dev/nvidiactl").exists():
        return True
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return False
    try:
        probe = subprocess.run(
            [nvidia_smi, "-L"],
            capture_output=True,
            text=True,
            timeout=1,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    if probe.returncode != 0:
        return False
    return bool(str(probe.stdout or "").strip())


def probe_gpu_inventory() -> GpuInventory:
    if not _gpu_host_candidate_exists():
        return GpuInventory(devices=())
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
