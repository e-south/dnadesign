"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/notebook_render_contract.py

DenseGen notebook-to-BaseRender contract for records rendering defaults.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

REQUIRED_ADAPTER_COLUMN_KEYS = ("id", "sequence", "annotations")


@dataclass(frozen=True)
class DenseGenNotebookRenderContract:
    adapter_kind: str
    adapter_columns: Mapping[str, str]
    adapter_policies: Mapping[str, object]
    style_preset: str
    record_window_limit: int


def _validate_notebook_render_contract(contract: DenseGenNotebookRenderContract) -> None:
    if not isinstance(contract.adapter_kind, str) or not contract.adapter_kind.strip():
        raise ValueError("adapter_kind must be a non-empty string")
    if not isinstance(contract.style_preset, str) or not contract.style_preset.strip():
        raise ValueError("style_preset must be a non-empty string")
    missing_keys = [key for key in REQUIRED_ADAPTER_COLUMN_KEYS if key not in contract.adapter_columns]
    if missing_keys:
        raise ValueError(f"adapter_columns missing required keys: {missing_keys}")
    for key in REQUIRED_ADAPTER_COLUMN_KEYS:
        value = contract.adapter_columns[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"adapter_columns[{key!r}] must be a non-empty string")
    if int(contract.record_window_limit) <= 0:
        raise ValueError("record_window_limit must be > 0")


def densegen_notebook_render_contract() -> DenseGenNotebookRenderContract:
    contract = DenseGenNotebookRenderContract(
        adapter_kind="densegen_tfbs",
        adapter_columns={
            "sequence": "sequence",
            "annotations": "densegen__used_tfbs_detail",
            "id": "id",
        },
        adapter_policies={},
        style_preset="presentation_default",
        record_window_limit=500,
    )
    _validate_notebook_render_contract(contract)
    return contract
