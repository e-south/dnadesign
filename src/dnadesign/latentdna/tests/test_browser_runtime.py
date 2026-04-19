"""Notebook runtime assembly helpers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dnadesign.latentdna.src.contracts.errors import ContractViolationError
from dnadesign.latentdna.src.contracts.plot_semantics import PlotSemantics
from dnadesign.latentdna.src.notebooks.browser_runtime import (
    _parse_deliverable_markdown,
    _plot_review_sections,
    resolve_plot_doc_block,
    resolve_runtime_hue_kinds,
)


def test_resolve_plot_doc_block_uses_plot_specific_subsection() -> None:
    markdown = """# Design-structure summary

Short deliverable summary.

## Why this deliverable exists

Deliverable context.

## Plot guide

Guide text.

### design_structure_summary | Design-structure summary

#### Plot details

**Data.** Design-structure details.

**Definition.** Read the separation-ratio panels directly.
"""

    parsed = _parse_deliverable_markdown(markdown)
    block = resolve_plot_doc_block(
        plot_id="design_structure_summary",
        deliverable_summary="Fallback summary.",
        parsed_markdown=parsed,
    )

    assert parsed["summary_markdown"] == "Short deliverable summary."
    assert block["title"] == "Design-structure summary"
    assert "Design-structure details." in str(block["markdown"])
    assert "**Data.** Design-structure details." in block["plot_details_md"]
    assert block["warning"] is None


def test_resolve_plot_doc_block_warns_when_subsection_is_missing() -> None:
    markdown = """# Context robustness summary

Deliverable fallback summary.

## Why this deliverable exists

Deliverable context.
"""

    parsed = _parse_deliverable_markdown(markdown)
    block = resolve_plot_doc_block(
        plot_id="context_robustness_summary",
        deliverable_summary="Fallback summary from deliverable contract.",
        parsed_markdown=parsed,
    )

    assert block["title"] == "Context robustness summary"
    assert block["markdown"] == "Deliverable fallback summary."
    assert block["plot_details_md"] == "Deliverable fallback summary."
    assert block["warning"] == "Missing plot-specific study-doc subsection for `context_robustness_summary`."


def test_resolve_runtime_hue_kinds_preserves_binary_entries() -> None:
    assert resolve_runtime_hue_kinds(
        ["design_family", "is_control", "context_shift_l2", "ignored_metric"],
        {
            "design_family": "categorical",
            "is_control": "binary",
            "context_shift_l2": "continuous",
            "ignored_metric": "unknown",
        },
    ) == {
        "design_family": "categorical",
        "is_control": "binary",
        "context_shift_l2": "continuous",
    }


def test_plot_review_sections_fail_closed_when_manifest_lacks_semantics(monkeypatch, tmp_path: Path) -> None:
    plot_dir = tmp_path / "outputs" / "plots" / "dataset_overview"
    plot_dir.mkdir(parents=True)
    (plot_dir / "manifest.json").write_text(
        json.dumps({"artifact_id": "dataset_overview", "outputs": [], "status": "ok"}),
        encoding="utf-8",
    )
    context = SimpleNamespace(
        config=SimpleNamespace(deliverables={}, plots={"dataset_overview": object()}),
        require_plot=lambda plot_id: SimpleNamespace(visibility_tier="primary", semantics_ref=None),
        workspace_dir=tmp_path,
    )
    monkeypatch.setattr(
        "dnadesign.latentdna.src.notebooks.browser_runtime._resolve_review_plot_spec",
        lambda context, *, plot_id: SimpleNamespace(kind="categorical_count", model_dump=lambda mode="json": {}),
    )

    with pytest.raises(ContractViolationError, match="semantics_ref"):
        _plot_review_sections(
            context,
            output_root=tmp_path / "outputs",
            controls={
                "plot_controls": {
                    "ordered_plot_ids": ["dataset_overview"],
                    "plots": [{"plot_id": "dataset_overview", "deliverable_id": "", "status": "ok"}],
                }
            },
        )


def test_plot_review_sections_preserve_plot_status_from_catalog(monkeypatch, tmp_path: Path) -> None:
    plot_dir = tmp_path / "outputs" / "plots" / "dataset_overview"
    plot_dir.mkdir(parents=True)
    (plot_dir / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_id": "dataset_overview",
                "status": "ok",
                "outputs": [],
                "semantics": {"caption": "legacy partial payload"},
            }
        ),
        encoding="utf-8",
    )
    context = SimpleNamespace(
        config=SimpleNamespace(deliverables={}, plots={"dataset_overview": object()}),
        require_plot=lambda plot_id: SimpleNamespace(visibility_tier="primary", semantics_ref="unused"),
        workspace_dir=tmp_path,
    )
    monkeypatch.setattr(
        "dnadesign.latentdna.src.notebooks.browser_runtime._resolve_review_plot_spec",
        lambda context, *, plot_id: SimpleNamespace(kind="categorical_count", model_dump=lambda mode="json": {}),
    )
    monkeypatch.setattr(
        "dnadesign.latentdna.src.notebooks.browser_runtime.resolve_plot_semantics",
        lambda context, *, plot_id: PlotSemantics(
            plot_id=plot_id,
            question="Does the plot stay current?",
            decision_role="primary",
            encoding="Fixture encoding.",
            scope="Fixture scope.",
            guardrails=["Fixture guardrail."],
            caption="Validated caption.",
            alt_text="Validated alt text.",
            preprocessing_md="Validated preprocessing.",
            math_md="Validated math.",
            rationale_md="Validated rationale.",
            limitations_md="Validated limitations.",
            failure_modes_md="Validated failure modes.",
        ),
    )

    review = _plot_review_sections(
        context,
        output_root=tmp_path / "outputs",
        controls={
            "plot_controls": {
                "ordered_plot_ids": ["dataset_overview"],
                "plots": [{"plot_id": "dataset_overview", "deliverable_id": "", "status": "attention", "stale": True}],
            }
        },
    )

    assert review.sections[0]["cards"][0]["status"] == "attention"
    assert review.sections[0]["cards"][0]["stale"] is True
    assert review.sections[0]["cards"][0]["caption_md"] == "Validated caption."


def test_plot_review_sections_revalidate_manifest_semantics_against_sidecar(
    monkeypatch,
    tmp_path: Path,
) -> None:
    plot_dir = tmp_path / "outputs" / "plots" / "dataset_overview"
    plot_dir.mkdir(parents=True)
    (plot_dir / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_id": "dataset_overview",
                "status": "ok",
                "outputs": [],
                "semantics": {"plot_id": "wrong_plot", "caption": "stale"},
            }
        ),
        encoding="utf-8",
    )
    context = SimpleNamespace(
        config=SimpleNamespace(deliverables={}, plots={"dataset_overview": object()}),
        require_plot=lambda plot_id: SimpleNamespace(visibility_tier="primary", semantics_ref="unused"),
        workspace_dir=tmp_path,
    )
    monkeypatch.setattr(
        "dnadesign.latentdna.src.notebooks.browser_runtime._resolve_review_plot_spec",
        lambda context, *, plot_id: SimpleNamespace(kind="categorical_count", model_dump=lambda mode="json": {}),
    )
    monkeypatch.setattr(
        "dnadesign.latentdna.src.notebooks.browser_runtime.resolve_plot_semantics",
        lambda context, *, plot_id: PlotSemantics(
            plot_id=plot_id,
            question="Does the plot stay current?",
            decision_role="primary",
            encoding="Fixture encoding.",
            scope="Fixture scope.",
            guardrails=["Fixture guardrail."],
            caption="Sidecar caption.",
            alt_text="Sidecar alt text.",
            preprocessing_md="Sidecar preprocessing.",
            math_md="Sidecar math.",
            rationale_md="Sidecar rationale.",
            limitations_md="Sidecar limitations.",
            failure_modes_md="Sidecar failure modes.",
        ),
    )

    review = _plot_review_sections(
        context,
        output_root=tmp_path / "outputs",
        controls={
            "plot_controls": {
                "ordered_plot_ids": ["dataset_overview"],
                "plots": [{"plot_id": "dataset_overview", "deliverable_id": "", "status": "ok"}],
            }
        },
    )

    assert review.sections[0]["cards"][0]["caption_md"] == "Sidecar caption."
