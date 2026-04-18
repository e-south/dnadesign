"""Workspace-side loading for decoupled plot semantics sidecars."""

from __future__ import annotations

from pathlib import Path

import yaml

from ..contracts.errors import ContractViolationError
from ..contracts.plot_semantics import PlotSemantics
from .loader import WorkspaceContext


def _semantics_path(context: WorkspaceContext, *, semantics_ref: str) -> Path:
    candidate = Path(semantics_ref)
    if not candidate.is_absolute():
        candidate = context.workspace_dir / candidate
    return candidate.resolve()


def _generated_fallback(plot_id: str) -> PlotSemantics:
    pretty = plot_id.replace("_", " ")
    return PlotSemantics(
        plot_id=plot_id,
        research_question=f"QC view for {pretty}.",
        evidence_tier="qc",
        encoding_summary=f"Mechanically generated QC semantics for {pretty}.",
        sampling_scope="Scope not declared.",
        interpretation_guardrails=["Fallback semantics are descriptive only."],
        caption_md=f"QC-only plot for {pretty}.",
        alt_text=f"QC-only plot for {pretty}.",
    )


def resolve_plot_semantics(
    context: WorkspaceContext,
    *,
    plot_id: str,
    allow_generated_fallback: bool = False,
) -> PlotSemantics:
    if allow_generated_fallback and plot_id not in context.config.plots:
        return _generated_fallback(plot_id)
    plot = context.require_plot(plot_id)
    semantics_ref = getattr(plot, "semantics_ref", None)
    if not semantics_ref:
        if allow_generated_fallback:
            return _generated_fallback(plot_id)
        raise ContractViolationError(f"persisted plot {plot_id!r} is missing semantics_ref")
    semantics_path = _semantics_path(context, semantics_ref=semantics_ref)
    if not semantics_path.is_file():
        raise ContractViolationError(f"plot semantics sidecar does not exist for {plot_id!r}: {semantics_ref}")
    payload = yaml.safe_load(semantics_path.read_text(encoding="utf-8")) or {}
    semantics = PlotSemantics.model_validate(payload)
    if semantics.plot_id != plot_id:
        raise ContractViolationError(f"plot semantics sidecar plot_id mismatch for {plot_id!r}: {semantics.plot_id!r}")
    return semantics


def validate_plot_semantics_sidecars(context: WorkspaceContext) -> None:
    for plot_id in sorted(context.config.plots):
        resolve_plot_semantics(context, plot_id=plot_id)
