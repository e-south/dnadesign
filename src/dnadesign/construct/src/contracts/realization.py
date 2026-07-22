"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/contracts/realization.py

Template realization and window extraction contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import Field, field_validator, model_validator

from .base import StrictConfigModel


class WindowConfig(StrictConfigModel):
    semantics: Literal["fixed_total", "anchor_plus_context"] = "fixed_total"
    reference: Literal["start", "center", "end"] = "center"
    direction: Literal["symmetric", "five_prime", "three_prime"] = "symmetric"
    size_bp: Optional[int] = Field(default=None, ge=1)
    upstream_bp: Optional[int] = Field(default=None, ge=0)
    downstream_bp: Optional[int] = Field(default=None, ge=0)
    offset_bp: int = 0

    @model_validator(mode="after")
    def _validate_shape(self) -> "WindowConfig":
        if self.semantics == "fixed_total":
            if self.size_bp is None:
                raise ValueError("realize.window.size_bp is required when realize.window.semantics='fixed_total'.")
            if self.upstream_bp is not None or self.downstream_bp is not None:
                raise ValueError(
                    "realize.window.upstream_bp/downstream_bp are only allowed when "
                    "realize.window.semantics='anchor_plus_context'."
                )
            return self

        if self.size_bp is not None:
            raise ValueError("realize.window.size_bp is only allowed when realize.window.semantics='fixed_total'.")
        if self.upstream_bp is None or self.downstream_bp is None:
            raise ValueError(
                "realize.window.upstream_bp and realize.window.downstream_bp are required when "
                "realize.window.semantics='anchor_plus_context'."
            )
        if self.direction != "symmetric":
            raise ValueError(
                "realize.window.direction must stay 'symmetric' when semantics='anchor_plus_context'. "
                "Use upstream_bp/downstream_bp to express asymmetric flanks."
            )
        if self.reference != "center":
            raise ValueError(
                "realize.window.reference must stay 'center' when semantics='anchor_plus_context'. "
                "The extracted span is defined by the focal part plus explicit upstream/downstream flanks."
            )
        if self.offset_bp != 0:
            raise ValueError("realize.window.offset_bp is only supported when semantics='fixed_total'.")
        return self


class RealizeConfig(StrictConfigModel):
    mode: Literal["window", "full_construct"] = "window"
    focal_part: Optional[str] = None
    required_slots: List[str] = Field(default_factory=list)
    window: Optional[WindowConfig] = None

    @field_validator("required_slots")
    @classmethod
    def _required_slots_not_blank(cls, value: List[str]) -> List[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for item in value:
            text = str(item or "").strip()
            if not text:
                raise ValueError("realize.required_slots cannot contain empty strings.")
            if text in seen:
                raise ValueError(f"realize.required_slots contains duplicate slot '{text}'.")
            seen.add(text)
            normalized.append(text)
        return normalized

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_window_fields(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        legacy_fields = [field for field in ("focal_point", "anchor_offset_bp", "window_bp") if field in data]
        if legacy_fields:
            joined = ", ".join(f"realize.{field}" for field in legacy_fields)
            raise ValueError(
                f"{joined} is no longer supported. Use realize.window.reference, "
                "realize.window.offset_bp, and realize.window.size_bp instead."
            )
        if data.get("mode", "window") == "window" and "window" in data and data["window"] is None:
            raise ValueError("realize.window must be a mapping when realize.mode='window'.")
        return data

    @model_validator(mode="after")
    def _validate_mode(self) -> "RealizeConfig":
        if self.mode == "window":
            if not str(self.focal_part or "").strip():
                raise ValueError("realize.focal_part is required when realize.mode='window'.")
            if self.window is None:
                raise ValueError("realize.window is required when realize.mode='window'.")
            return self
        if self.window is not None:
            raise ValueError("realize.window is only allowed when realize.mode='window'.")
        return self
