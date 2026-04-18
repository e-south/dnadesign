"""Plot-level semantics contracts for persisted LatentDNA figures."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PlotSemantics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plot_id: str = Field(min_length=1)
    research_question: str = Field(min_length=1)
    evidence_tier: Literal["primary", "secondary", "appendix", "qc"]
    encoding_summary: str = Field(min_length=1)
    sampling_scope: str = Field(min_length=1)
    interpretation_guardrails: list[str] = Field(min_length=1)
    caption_md: str = Field(min_length=1)
    alt_text: str = Field(min_length=1)
    docs_refs: list[str] = Field(default_factory=list)
