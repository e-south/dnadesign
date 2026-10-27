"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/contracts/datasets.py

USR dataset locator and input selection contracts for construct configs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import field_validator, model_validator

from .base import StrictConfigModel


class USRDatasetLocatorConfig(StrictConfigModel):
    kind: Literal["usr"]
    dataset: str
    root: Optional[str] = None

    @field_validator("dataset")
    @classmethod
    def _dataset_not_blank(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("USR dataset locator dataset cannot be empty.")
        return text


class InputConfig(StrictConfigModel):
    source: USRDatasetLocatorConfig
    field: Optional[str] = "sequence"
    ids: Optional[List[str]] = None

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_shape(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        legacy_fields = [field for field in ("dataset", "root") if field in data]
        if isinstance(data.get("source"), str):
            legacy_fields.append("source")
        if legacy_fields:
            joined = ", ".join(f"input.{field}" for field in legacy_fields)
            raise ValueError(
                f"{joined} is no longer supported. Use input.source.kind, "
                "input.source.dataset, and input.source.root instead."
            )
        return data

    @field_validator("field")
    @classmethod
    def _field_not_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value or "").strip()
        if not text:
            raise ValueError("input.field cannot be empty.")
        return text
