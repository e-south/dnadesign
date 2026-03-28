"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/tests/runtime/test_hardware_probe.py

Contracts for infer GPU inventory probing.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

from dnadesign.infer.src.runtime import hardware_probe


def test_probe_gpu_inventory_fast_paths_to_empty_without_gpu_host_candidate(monkeypatch) -> None:
    monkeypatch.setattr(hardware_probe, "_gpu_host_candidate_exists", lambda: False)

    inventory = hardware_probe.probe_gpu_inventory()

    assert inventory.count == 0
    assert inventory.devices == ()


def test_probe_gpu_inventory_uses_torch_when_gpu_host_candidate_exists(monkeypatch) -> None:
    monkeypatch.setattr(hardware_probe, "_gpu_host_candidate_exists", lambda: True)
    fake_props = SimpleNamespace(name="Demo GPU", total_memory=8 * 1024**3, major=9, minor=0)
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            get_device_properties=lambda idx: fake_props,
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    inventory = hardware_probe.probe_gpu_inventory()

    assert inventory.count == 1
    assert inventory.devices[0].name == "Demo GPU"
    assert inventory.devices[0].total_memory_gib == 8.0
    assert inventory.devices[0].compute_capability == "9.0"
