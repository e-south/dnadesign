"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/__init__.py

Public API:
  - run_extract
  - run_evo2_sequence_features
  - run_generate
  - run_job (YAML-driven)
  - export_evo2_sequence_opal_matrix
  - inspect_local_gpu_inventory
  - resolve_infer_runtime_lane_contracts
  - validate_runbook_gpu_resources

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .contracts import (
    InferRuntimeLaneContract,
    infer_model_family_suffix,
    plan_sequence_view_feature_completion_from_config,
    plan_sequence_view_feature_inventory_completion_from_config,
    resolve_infer_runtime_lane_contracts,
    validate_infer_config_contract,
    validate_infer_dry_run_contract,
)


def run_extract(*args: Any, **kwargs: Any):
    from .src.api import run_extract as _run_extract

    return _run_extract(*args, **kwargs)


def run_generate(*args: Any, **kwargs: Any):
    from .src.api import run_generate as _run_generate

    return _run_generate(*args, **kwargs)


def run_evo2_sequence_features(*args: Any, **kwargs: Any):
    from .src.api import run_evo2_sequence_features as _run_evo2_sequence_features

    return _run_evo2_sequence_features(*args, **kwargs)


def run_job(*args: Any, **kwargs: Any):
    from .src.api import run_job as _run_job

    return _run_job(*args, **kwargs)


def export_evo2_sequence_opal_matrix(*args: Any, **kwargs: Any):
    from .src.api import export_evo2_sequence_opal_matrix as _export_evo2_sequence_opal_matrix

    return _export_evo2_sequence_opal_matrix(*args, **kwargs)


def inspect_local_gpu_inventory() -> dict[str, object]:
    from .src.runtime.hardware_probe import probe_gpu_inventory

    inventory = probe_gpu_inventory()
    return {
        "count": inventory.count,
        "devices": [
            {
                "index": device.index,
                "name": device.name,
                "total_memory_gib": float(device.total_memory_gib),
                "compute_capability": device.compute_capability,
            }
            for device in inventory.devices
        ],
    }


def validate_runbook_gpu_resources(
    *,
    config_path: Path,
    declared_gpus: int,
    gpu_capability: str | None,
    gpu_memory_gib: float | None,
) -> None:
    from .src.resource_contracts import validate_runbook_gpu_resources as _validate_runbook_gpu_resources

    _validate_runbook_gpu_resources(
        config_path=config_path,
        declared_gpus=declared_gpus,
        gpu_capability=gpu_capability,
        gpu_memory_gib=gpu_memory_gib,
    )


__all__ = (
    "run_extract",
    "run_evo2_sequence_features",
    "run_generate",
    "run_job",
    "export_evo2_sequence_opal_matrix",
    "InferRuntimeLaneContract",
    "infer_model_family_suffix",
    "inspect_local_gpu_inventory",
    "plan_sequence_view_feature_completion_from_config",
    "plan_sequence_view_feature_inventory_completion_from_config",
    "resolve_infer_runtime_lane_contracts",
    "validate_infer_config_contract",
    "validate_infer_dry_run_contract",
    "validate_runbook_gpu_resources",
)
