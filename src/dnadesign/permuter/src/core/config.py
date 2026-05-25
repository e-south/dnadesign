"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/core/config.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from dnadesign.permuter.src.plots.registry import supported_plot_ids


class PermuterModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --- Workspace scope YAML schema ---------------------------------------------------------


class ScopeInput(PermuterModel):
    refs: str
    name_col: str = "ref_name"
    seq_col: str = "sequence"
    aa_col: Optional[str] = None
    reference_sequence: Optional[str] = None


class ScopePermute(PermuterModel):
    protocol: str = Field(description="Protocol id (e.g., scan_dna|scan_codon|scan_stem_loop)")
    params: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("protocol")
    @classmethod
    def _known_protocol(cls, v: str):
        allowed = {
            "scan_dna",
            "scan_codon",
            "scan_stem_loop",
            "combine_aa",
            "multisite_select",
        }
        if v not in allowed:
            raise ValueError(f"Unknown protocol: {v!r}. Allowed: {sorted(allowed)}")
        return v


class ScopeOutput(PermuterModel):
    dir: str
    layout: Optional[str] = None


class ScopePlot(PermuterModel):
    which: List[str] = Field(default_factory=lambda: ["position_scatter_and_heatmap"])
    metric_id: Optional[str] = None
    # Draw every Nth AA in the reference strip (None → auto ≈ 200 labels total)
    strip_every: Optional[int] = Field(default=None, ge=1, le=50)
    emit_summaries: bool = True

    # Optional figure size in inches (matplotlib figsize)
    class PlotSize(PermuterModel):
        width: Optional[float] = Field(default=None, ge=2.0, le=64.0)
        height: Optional[float] = Field(default=None, ge=2.0, le=64.0)

    size: Optional[PlotSize] = None
    sizes: Dict[str, PlotSize] = Field(default_factory=dict)
    # Multiplicative font scaling factor applied to plot text
    font_scale: float = Field(default=1.0, ge=0.5, le=3.0)
    ranked_jitter: Optional[float] = None
    ranked_point_size: Optional[float] = None
    ranked_alpha: Optional[float] = None
    ranked_cmap: Optional[str] = None
    ranked_annotate_top: Optional[int] = None
    ranked_summary_top_n: Optional[int] = None
    ranked_export_top_k: Optional[int] = None
    ranked_xtick_every: Optional[int] = None

    @field_validator("which")
    @classmethod
    def _allowed_plots(cls, vs: List[str]):
        allowed = set(supported_plot_ids())
        bad = [x for x in vs if x not in allowed]
        if bad:
            raise ValueError(f"Unknown plot(s): {bad}. Allowed: {sorted(allowed)}")
        return vs

    @field_validator("sizes")
    @classmethod
    def _sizes_keys_valid(cls, v: Dict[str, "ScopePlot.PlotSize"]):
        allowed = set(supported_plot_ids())
        bad = [k for k in v.keys() if k not in allowed]
        if bad:
            raise ValueError(f"plot.sizes has invalid key(s): {bad}. Allowed: {sorted(allowed)}")
        return v


class EvalMetric(PermuterModel):
    id: str  # column suffix → permuter__observed__<id>
    evaluator: str  # registry key, e.g. evo2_llr
    metric: str  # evaluator's internal metric name (e.g. "log_likelihood_ratio")
    params: Dict[str, Any] = Field(default_factory=dict)


class ScopeEvaluate(PermuterModel):
    metrics: List[EvalMetric]

    @field_validator("metrics")
    @classmethod
    def _unique_metric_ids(cls, v: List[EvalMetric]):
        ids = [m.id for m in v]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate metric id(s) in evaluate.metrics: {ids}")
        return v


class ScopeDefinition(PermuterModel):
    name: str
    input: ScopeInput
    permute: ScopePermute
    output: ScopeOutput
    evaluate: Optional[ScopeEvaluate] = None
    plot: Optional[ScopePlot] = None
    bio_type: Optional[Literal["dna", "protein"]] = None


class ScopeConfig(PermuterModel):
    scope: ScopeDefinition

    def infer_bio_type(self, sequence_hint: str | None = None) -> Literal["dna", "protein"]:
        if self.scope.bio_type in ("dna", "protein"):
            return self.scope.bio_type  # explicit
        s = (sequence_hint or "").upper()
        # if strictly A/C/G/T → dna; otherwise assume protein
        return "dna" if s and all(ch in "ACGT" for ch in s) else "protein"
