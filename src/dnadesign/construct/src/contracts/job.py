"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/contracts/job.py

Top-level construct job contracts and cross-field invariants.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import Field, model_validator

from .base import StrictConfigModel
from .datasets import InputConfig
from .job_invariants import (
    require_output_variant_anchor_handoff_contract,
    require_realize_focal_contract,
)
from .normalize_anchor import NormalizeAnchorConfig
from .output import OutputConfig, OutputVariantConfig
from .parts import PartConfig
from .realization import RealizeConfig
from .templates import TemplateConfig


class InnerJobConfig(StrictConfigModel):
    id: str
    mode: Literal["realize_template", "normalize_anchor"] = "realize_template"
    input: InputConfig
    template: Optional[TemplateConfig] = None
    parts: List[PartConfig] = Field(default_factory=list)
    realize: Optional[RealizeConfig] = None
    normalize_anchor: Optional[NormalizeAnchorConfig] = None
    output_variants: List[OutputVariantConfig] = Field(default_factory=list)
    output: OutputConfig

    @model_validator(mode="after")
    def _validate_parts(self) -> "InnerJobConfig":
        if not str(self.input.source.root or "").strip():
            raise ValueError("job.input.source.root is required for construct jobs that read USR datasets.")
        if self.mode == "normalize_anchor":
            if self.input.field is None:
                raise ValueError("job.input.field is required when job.mode='normalize_anchor'.")
            if self.normalize_anchor is None:
                raise ValueError("job.normalize_anchor is required when job.mode='normalize_anchor'.")
            if self.template is not None:
                raise ValueError("job.template is only allowed when job.mode='realize_template'.")
            if self.parts:
                raise ValueError("job.parts is only allowed when job.mode='realize_template'.")
            if self.realize is not None:
                raise ValueError("job.realize is only allowed when job.mode='realize_template'.")
            if self.output_variants:
                raise ValueError("job.output_variants is only allowed when job.mode='realize_template'.")
            return self
        if self.template is None:
            raise ValueError("job.template is required when job.mode='realize_template'.")
        if not self.parts:
            raise ValueError("job.parts must define at least one part.")
        if self.realize is None:
            raise ValueError("job.realize is required when job.mode='realize_template'.")
        if self.normalize_anchor is not None:
            raise ValueError("job.normalize_anchor is only allowed when job.mode='normalize_anchor'.")
        seen: set[str] = set()
        input_driven = 0
        for part in self.parts:
            if part.name in seen:
                raise ValueError(f"Duplicate part name '{part.name}'.")
            seen.add(part.name)
            if part.sequence.source == "input_field":
                input_driven += 1
        if input_driven < 1:
            raise ValueError("job.parts must include at least one source='input_field' part.")
        require_realize_focal_contract(
            parts=self.parts,
            focal_part=self.realize.focal_part,
            realize_mode=self.realize.mode,
        )
        require_output_variant_anchor_handoff_contract(
            parts=self.parts,
            focal_part=self.realize.focal_part,
            output_variants=self.output_variants,
        )
        missing_required_slots = [slot for slot in self.realize.required_slots if slot not in seen]
        if missing_required_slots:
            joined = ", ".join(missing_required_slots)
            raise ValueError(f"realize.required_slots contains unknown part name(s): {joined}.")
        return self


class JobConfig(StrictConfigModel):
    job: InnerJobConfig
