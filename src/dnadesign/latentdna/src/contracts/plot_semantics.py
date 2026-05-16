"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/contracts/plot_semantics.py

Plot-level semantics contracts for persisted LatentDNA figures.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PlotSemantics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plot_id: str = Field(min_length=1)
    title: str | None = Field(default=None, min_length=1)
    question: str = Field(min_length=1)
    decision_role: Literal["gate", "primary", "appendix", "debug"]
    encoding: str = Field(min_length=1)
    scope: str = Field(min_length=1)
    guardrails: list[str] = Field(min_length=1)
    caption: str = Field(min_length=1)
    alt_text: str = Field(min_length=1)
    preprocessing_md: str = Field(min_length=1)
    math_md: str = Field(min_length=1)
    rationale_md: str = Field(min_length=1)
    plot_details_md: str | None = None
    limitations_md: str = Field(min_length=1)
    failure_modes_md: str = Field(min_length=1)
    docs_refs: list[str] = Field(default_factory=list)
