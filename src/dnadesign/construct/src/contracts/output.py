"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/contracts/output.py

Output dataset and emitted variant contracts for construct configs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import field_validator, model_validator

from .base import StrictConfigModel
from .datasets import USRDatasetLocatorConfig


class OutputVariantConfig(StrictConfigModel):
    product_kind: Literal["realized_context"]
    orientation: Literal["forward", "reverse_complement"]
    recommended_pooling: Optional[Literal["seq_mean", "anchor_mean", "core60_mean"]] = None
    anchor_part: Optional[str] = None
    view_name: Optional[str] = None

    @model_validator(mode="after")
    def _validate_product_kind_orientation(self) -> "OutputVariantConfig":
        if self.product_kind != "realized_context":
            raise ValueError("output_variants product_kind must be 'realized_context'.")
        return self

    @field_validator("anchor_part", "view_name")
    @classmethod
    def _optional_text_not_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value or "").strip()
        if not text:
            raise ValueError("output_variants anchor_part/view_name cannot be empty when provided.")
        return text


class OutputConfig(StrictConfigModel):
    target: USRDatasetLocatorConfig
    record_source: Optional[str] = None
    on_conflict: Literal["error", "ignore"] = "error"
    allow_same_as_input: bool = False

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_shape(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        legacy_fields = [field for field in ("dataset", "root", "source") if field in data]
        if legacy_fields:
            joined = ", ".join(f"output.{field}" for field in legacy_fields)
            raise ValueError(
                f"{joined} is no longer supported. Use output.target.kind, output.target.dataset, "
                "output.target.root, and output.record_source instead."
            )
        return data

    @field_validator("record_source")
    @classmethod
    def _record_source_not_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value or "").strip()
        if not text:
            raise ValueError("output.record_source cannot be empty when provided.")
        return text
