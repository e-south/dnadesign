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


def inline_plot_semantics(plot_id: str) -> PlotSemantics:
    pretty = plot_id.replace("_", " ")
    return PlotSemantics(
        plot_id=plot_id,
        question=f"QC view for {pretty}.",
        decision_role="debug",
        encoding=f"Inline QC render semantics for {pretty}.",
        scope="Scope not declared.",
        guardrails=["Inline plot semantics are descriptive only and are not a study-facing contract."],
        caption=f"QC-only plot for {pretty}.",
        alt_text=f"QC-only plot for {pretty}.",
        preprocessing_md="Inline semantics do not declare preprocessing.",
        math_md="Inline semantics do not declare a mathematical definition.",
        rationale_md="Inline semantics exist only for one-off QC rendering.",
        limitations_md="Inline semantics are not a study-facing scientific contract.",
        failure_modes_md="Declare plot semantics in workspace config before using this plot in a study surface.",
    )


def resolve_plot_semantics(
    context: WorkspaceContext,
    *,
    plot_id: str,
) -> PlotSemantics:
    plot = context.require_plot(plot_id)
    semantics_ref = getattr(plot, "semantics_ref", None)
    if not semantics_ref:
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
