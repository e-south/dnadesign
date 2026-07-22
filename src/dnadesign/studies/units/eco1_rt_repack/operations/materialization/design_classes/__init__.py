"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/design_classes/__init__.py

Public API for Eco1 RT design-class expansion materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "materialize_design_class_candidate_pool": (
        "dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes.candidate_pool",
        "materialize_design_class_candidate_pool",
    ),
    "materialize_design_class_downstream_inputs": (
        "dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes.downstream_inputs",
        "materialize_design_class_downstream_inputs",
    ),
    "materialize_design_class_esmc_sequence_preference": (
        "dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes.esmc_sequence_preference",
        "materialize_design_class_esmc_sequence_preference",
    ),
    "materialize_design_class_foldcheck_request": (
        "dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes.foldcheck",
        "materialize_design_class_foldcheck_request",
    ),
    "materialize_design_class_requests": (
        "dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes.pipeline",
        "materialize_design_class_requests",
    ),
}

__all__ = [
    "materialize_design_class_candidate_pool",
    "materialize_design_class_downstream_inputs",
    "materialize_design_class_esmc_sequence_preference",
    "materialize_design_class_foldcheck_request",
    "materialize_design_class_requests",
]


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _EXPORTS[name]
    return getattr(import_module(module_name), attribute_name)
