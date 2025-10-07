"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/core/config.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

# --- Job yaml schema ---------------------------------------------------------


class JobInput(BaseModel):
    refs: str
    name_col: str = "ref_name"
    seq_col: str = "sequence"
    aa_col: Optional[str] = None  # optional: protein sequence column


class JobPermute(BaseModel):
    protocol: str = Field(
        description="Protocol id (e.g., scan_dna|scan_codon|scan_stem_loop)"
    )
    params: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("protocol")
    @classmethod
    def _known_protocol(cls, v: str):
        allowed = {"scan_dna", "scan_codon", "scan_stem_loop"}
        if v not in allowed:
            raise ValueError(f"Unknown protocol: {v!r}. Allowed: {sorted(allowed)}")
        return v


class JobOutput(BaseModel):
    dir: str


class JobPlot(BaseModel):
    which: List[str] = Field(default_factory=lambda: ["position_scatter_and_heatmap"])
    metric_id: Optional[str] = None
    # Draw every Nth AA in the reference strip (None → auto ≈ 200 labels total)
    strip_every: Optional[int] = Field(default=None, ge=1, le=50)
    emit_summaries: bool = True

    # Optional figure size in inches (matplotlib figsize)
    class PlotSize(BaseModel):
        width: Optional[float] = Field(default=None, ge=2.0, le=64.0)
        height: Optional[float] = Field(default=None, ge=2.0, le=64.0)

    size: Optional[PlotSize] = None
    # Multiplicative font scaling factor applied to plot text
    font_scale: float = Field(default=1.0, ge=0.5, le=3.0)

    @field_validator("which")
    @classmethod
    def _allowed_plots(cls, vs: List[str]):
        allowed = {
            "position_scatter_and_heatmap",
            "metric_by_mutation_count",
            "aa_category_effects",
        }
        bad = [x for x in vs if x not in allowed]
        if bad:
            raise ValueError(f"Unknown plot(s): {bad}. Allowed: {sorted(allowed)}")
        return vs


class EvalMetric(BaseModel):
    id: str  # column suffix → permuter__metric__<id>
    evaluator: str  # registry key, e.g. evo2_llr
    metric: str  # evaluator's internal metric name (e.g. "log_likelihood_ratio")
    params: Dict[str, Any] = Field(default_factory=dict)


class JobEvaluate(BaseModel):
    metrics: List[EvalMetric]

    @field_validator("metrics")
    @classmethod
    def _unique_metric_ids(cls, v: List[EvalMetric]):
        ids = [m.id for m in v]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate metric id(s) in evaluate.metrics: {ids}")
        return v


class InnerJob(BaseModel):
    name: str
    input: JobInput
    permute: JobPermute
    output: JobOutput
    evaluate: Optional[JobEvaluate] = None
    plot: Optional[JobPlot] = None
    bio_type: Optional[Literal["dna", "protein"]] = None


class JobConfig(BaseModel):
    job: InnerJob

    def infer_bio_type(
        self, sequence_hint: str | None = None
    ) -> Literal["dna", "protein"]:
        if self.job.bio_type in ("dna", "protein"):
            return self.job.bio_type  # explicit
        s = (sequence_hint or "").upper()
        # if strictly A/C/G/T → dna; otherwise assume protein
        return "dna" if s and all(ch in "ACGT" for ch in s) else "protein"
