"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/config/postprocess.py

DenseGen postprocess configuration schema.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Literal


class PadGcConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["off", "range", "target"] = "range"
    min: float = 0.40
    max: float = 0.60
    target: float = 0.50
    tolerance: float = 0.10
    min_pad_length: int = 0

    @field_validator("min", "max", "target", "tolerance")
    @classmethod
    def _gc_ok(cls, v: float, info):
        if not (0.0 <= float(v) <= 1.0):
            raise ValueError(f"{info.field_name} must be between 0 and 1")
        return float(v)

    @field_validator("mode", mode="before")
    @classmethod
    def _coerce_mode(cls, v):
        if isinstance(v, bool):
            if v is False:
                return "off"
            raise ValueError("pad.gc.mode must be one of: off, range, target")
        return v

    @field_validator("min_pad_length")
    @classmethod
    def _min_pad_length_ok(cls, v: int):
        if int(v) < 0:
            raise ValueError("min_pad_length must be >= 0")
        return int(v)

    @model_validator(mode="after")
    def _gc_bounds(self):
        if self.min > self.max:
            raise ValueError("gc.min must be <= gc.max")
        if self.mode == "target":
            target_min = self.target - self.tolerance
            target_max = self.target + self.tolerance
            if target_min < 0.0 or target_max > 1.0:
                raise ValueError("gc.target +/- gc.tolerance must stay within [0, 1]")
        return self


class PadConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["off", "strict", "adaptive"] = "adaptive"
    end: Literal["5prime", "3prime"] = "5prime"
    gc: PadGcConfig = Field(default_factory=PadGcConfig)
    max_tries: int = 2000

    @field_validator("max_tries")
    @classmethod
    def _max_tries_ok(cls, v: int):
        if int(v) <= 0:
            raise ValueError("max_tries must be > 0")
        return int(v)

    @field_validator("mode", mode="before")
    @classmethod
    def _coerce_mode(cls, v):
        if isinstance(v, bool):
            if v is False:
                return "off"
            raise ValueError("pad.mode must be one of: off, strict, adaptive")
        return v


class FinalSequenceKmerFilterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kmers: List[str]

    @field_validator("kmers")
    @classmethod
    def _kmers_ok(cls, v: List[str]):
        if not v:
            raise ValueError(
                "postprocess.validate_final_sequence.forbid_kmers_outside_promoter_windows.kmers must be set"
            )
        cleaned: list[str] = []
        for raw in v:
            if not isinstance(raw, str):
                raise ValueError(
                    "postprocess.validate_final_sequence.forbid_kmers_outside_promoter_windows.kmers must be strings"
                )
            seq = raw.strip().upper()
            if not seq:
                raise ValueError(
                    "postprocess.validate_final_sequence.forbid_kmers_outside_promoter_windows.kmers must be non-empty"
                )
            if any(ch not in {"A", "C", "G", "T"} for ch in seq):
                raise ValueError(
                    "postprocess.validate_final_sequence.forbid_kmers_outside_promoter_windows.kmers "
                    "must contain only A/C/G/T"
                )
            cleaned.append(seq)
        if len(set(cleaned)) != len(cleaned):
            raise ValueError(
                "postprocess.validate_final_sequence.forbid_kmers_outside_promoter_windows.kmers must be unique"
            )
        return cleaned


class FinalSequenceValidationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    forbid_kmers_outside_promoter_windows: Optional[FinalSequenceKmerFilterConfig] = None


class PostprocessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pad: PadConfig = Field(default_factory=PadConfig)
    validate_final_sequence: Optional[FinalSequenceValidationConfig] = None
