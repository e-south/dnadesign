"""Notebook runtime assembly helpers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dnadesign.latentdna.src.contracts.errors import ContractViolationError
from dnadesign.latentdna.src.contracts.plot_semantics import PlotSemantics
from dnadesign.latentdna.src.notebooks.browser_runtime import (
    _parse_deliverable_markdown,
    _plot_review_sections,
    _reference_annotation_options,
    _reference_required_columns,
    _runtime_hue_columns,
    build_workspace_browser_runtime,
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
    assert block["markdown"] == ""
    assert "**Data.** Design-structure details." in block["plot_details_md"]
    assert block["warning"] is None


def test_resolve_plot_doc_block_preserves_non_plot_details_notes() -> None:
    markdown = """# Representation health summary

Short deliverable summary.

### representation_health_summary | Representation health summary

Use this panel to screen out collapsed candidate spaces before comparing design structure.

#### Plot details

**Data.** Candidate summary details.
"""

    parsed = _parse_deliverable_markdown(markdown)
    block = resolve_plot_doc_block(
        plot_id="representation_health_summary",
        deliverable_summary="Fallback summary.",
        parsed_markdown=parsed,
    )

    assert (
        block["markdown"]
        == "Use this panel to screen out collapsed candidate spaces before comparing design structure."
    )
    assert block["plot_details_md"] == "**Data.** Candidate summary details."


def test_resolve_plot_doc_block_stops_plot_section_at_parent_heading() -> None:
    markdown = """# Sigma-factor UMAP panel

Short deliverable summary.

### native_umap | Native UMAP

#### Plot details

Native UMAP details.

## Interpretation

Shared interpretation for all plots, not one plot's details.
"""

    parsed = _parse_deliverable_markdown(markdown)
    block = resolve_plot_doc_block(
        plot_id="native_umap",
        deliverable_summary="Fallback summary.",
        parsed_markdown=parsed,
    )

    assert block["plot_details_md"] == "Native UMAP details."
    assert "Shared interpretation" not in block["plot_details_md"]


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
    assert block["plot_details_md"] == ""
    assert block["warning"] == "Missing plot-specific study-doc subsection for `context_robustness_summary`."


def test_resolve_runtime_hue_kinds_preserves_binary_entries() -> None:
    assert resolve_runtime_hue_kinds(
        ["design_family", "is_control", "spacer_length", "context_shift_l2", "ignored_metric"],
        {
            "design_family": "categorical",
            "is_control": "binary",
            "spacer_length": "ordinal",
            "context_shift_l2": "continuous",
            "ignored_metric": "unknown",
        },
    ) == {
        "design_family": "categorical",
        "is_control": "binary",
        "spacer_length": "ordinal",
        "context_shift_l2": "continuous",
    }


def test_runtime_hue_columns_ignore_unbound_legacy_joinable_tables() -> None:
    global_hues, hue_kinds = _runtime_hue_columns(
        joinable_tables=[
            {
                "artifact_id": "context_delta_distribution_demo",
                "columns": ["id", "context_shift_l2"],
            }
        ],
        preferred_hues=["design_family", "context_shift_l2"],
        row_metadata_hues=[],
        configured_hue_kinds={
            "design_family": "categorical",
            "context_shift_l2": "continuous",
        },
        joinable_artifact_suffixes={"context_delta_distribution_demo"},
    )

    assert global_hues == []
    assert hue_kinds == {}


def test_runtime_hue_columns_accept_control_plane_view_row_metadata() -> None:
    global_hues, hue_kinds = _runtime_hue_columns(
        joinable_tables=[],
        preferred_hues=["source_family", "promoter_standard__strength_value_numeric"],
        row_metadata_hues=["source_family", "promoter_standard__strength_value_numeric"],
        configured_hue_kinds={
            "source_family": "categorical",
            "promoter_standard__strength_value_numeric": "continuous",
        },
        joinable_artifact_suffixes=set(),
    )

    assert global_hues == ["source_family", "promoter_standard__strength_value_numeric"]
    assert hue_kinds == {
        "source_family": "categorical",
        "promoter_standard__strength_value_numeric": "continuous",
    }


def test_browser_runtime_uses_control_plane_shapes_without_loading_matrices(monkeypatch, tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    output_root = workspace_dir / "outputs"
    notebook_dir = output_root / "notebooks" / "latent_geometry_browser"
    view_dir = output_root / "views" / "candidate_view"
    notebook_dir.mkdir(parents=True)
    view_dir.mkdir(parents=True)
    (view_dir / "matrix.npy").write_bytes(b"runtime should not inspect this matrix")
    catalog_path = output_root / "catalog.json"
    health_path = notebook_dir / "health.json"
    catalog_path.write_text(
        json.dumps(
            {
                "deliverables": [],
                "plots": [],
                "exports": [],
                "notebooks": [],
                "runs": [],
                "candidate_inventory": [
                    {
                        "study_id": "runtime_shape_demo",
                        "candidate_set_ids": ["demo_x"],
                        "view_id": "candidate_view",
                        "source_id": "features",
                        "dataset": "features.parquet",
                        "row_basis": "subject_id",
                        "model_name": "evo2_7b",
                        "feature_family": "intermediate_embedding",
                        "modality": "vector",
                        "sequence_scope": "anchor_60bp",
                        "pooling_operation": "seq_mean",
                        "orientation": "forward",
                        "coordinate_space_id": "demo_space",
                        "role": "primary",
                        "n_rows": 123,
                        "n_dims": 16,
                        "materialization_status": "materialized",
                        "freshness_status": "ok",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    health_path.write_text("{}", encoding="utf-8")
    controls = {
        "candidate_inventory": [
            {
                "study_id": "runtime_shape_demo",
                "candidate_set_ids": ["demo_x"],
                "view_id": "candidate_view",
                "source_id": "features",
                "dataset": "features.parquet",
                "row_basis": "subject_id",
                "model_name": "evo2_7b",
                "feature_family": "intermediate_embedding",
                "modality": "vector",
                "sequence_scope": "anchor_60bp",
                "pooling_operation": "seq_mean",
                "orientation": "forward",
                "coordinate_space_id": "demo_space",
                "role": "primary",
                "n_rows": 123,
                "n_dims": 16,
                "materialization_status": "materialized",
                "freshness_status": "ok",
            }
        ],
        "plot_controls": {"default_surface": "plots", "ordered_plot_ids": [], "plots": []},
        "geometry_controls": {
            "default_model": "7b",
            "default_family": "intermediate_embedding",
            "default_context": "anchor_60bp",
            "default_layout": "single_view",
            "geometries": [
                {
                    "view_id": "candidate_view",
                    "label": "Candidate view",
                    "model": "7b",
                    "family": "intermediate_embedding",
                    "context": "anchor_60bp",
                    "role": "primary",
                    "materialized": True,
                    "projection_ids": [],
                    "coordinate_space_id": "demo_space",
                    "rows": 123,
                    "dims": 16,
                }
            ],
            "preferred_hues": [],
            "row_metadata_hues": [],
            "hue_kinds": {},
            "joinable_tables": [],
            "layout_presets": [],
            "comparison_bases": [],
            "reference_labels": [],
            "reference_sets": [
                {
                    "reference_set_id": "reference_sfxi_archive",
                    "label": "SFXI archive",
                    "match_column": "usr_label__primary",
                    "label_column": "promoter_standard__display_name",
                    "label_mode": "label_and_highlight",
                    "label_limit": 32,
                    "explicit_ids": [],
                    "selector_columns": [],
                },
                {
                    "reference_set_id": "reference_native_mg1655",
                    "label": "Native MG1655 GenBank panel",
                    "match_column": "usr_label__primary",
                    "label_column": "promoter_standard__display_name",
                    "label_mode": "label_and_highlight",
                    "label_limit": 32,
                    "explicit_ids": [],
                    "selector_columns": [],
                },
            ],
            "reference_hue_options": [
                {
                    "label": "SFXI score",
                    "column": "sfxi_ref__sfxi",
                    "type": "continuous",
                    "reference_set_ids": ["reference_sfxi_archive"],
                },
                {
                    "label": "SFXI logic fidelity",
                    "column": "sfxi_ref__logic_fidelity",
                    "type": "continuous",
                    "reference_set_ids": ["reference_sfxi_archive"],
                },
                {
                    "label": "SFXI effect scaled",
                    "column": "sfxi_ref__effect_scaled",
                    "type": "continuous",
                    "reference_set_ids": ["reference_sfxi_archive"],
                },
            ],
            "reference_hue_options_by_reference_set": {
                "reference_sfxi_archive": [
                    {
                        "label": "SFXI score",
                        "column": "sfxi_ref__sfxi",
                        "type": "continuous",
                        "reference_set_ids": ["reference_sfxi_archive"],
                    },
                    {
                        "label": "SFXI logic fidelity",
                        "column": "sfxi_ref__logic_fidelity",
                        "type": "continuous",
                        "reference_set_ids": ["reference_sfxi_archive"],
                    },
                    {
                        "label": "SFXI effect scaled",
                        "column": "sfxi_ref__effect_scaled",
                        "type": "continuous",
                        "reference_set_ids": ["reference_sfxi_archive"],
                    },
                ],
                "reference_native_mg1655": [],
            },
            "candidate_sets": [],
            "compare_metrics": {},
        },
    }
    context = SimpleNamespace(
        config=SimpleNamespace(
            sources={},
            views={"candidate_view": SimpleNamespace(tags={"family": "intermediate_embedding"})},
            deliverables={},
            plots={},
        ),
        workspace_dir=workspace_dir,
    )
    monkeypatch.setattr("dnadesign.latentdna.src.notebooks.browser_runtime.load_workspace_config", lambda _: context)

    def _fail_matrix_load(*args, **kwargs):  # pragma: no cover - only runs on regression
        raise AssertionError("notebook runtime should not load matrix files for shape metadata")

    monkeypatch.setattr(np, "load", _fail_matrix_load)

    runtime = build_workspace_browser_runtime(
        title="Runtime shape demo",
        description=None,
        workspace_id="runtime_shape_demo",
        notebook_id="latent_geometry_browser",
        default_deliverable="demo",
        workspace_dir=workspace_dir,
        output_root=output_root,
        catalog_path=catalog_path,
        health_path=health_path,
        controls=controls,
    )

    assert runtime.identity.row_count_text == "candidate_view=123"
    assert runtime.identity.dimensionality_text == "candidate_view=16"
    assert runtime.support.reference_annotation_mode_options()["Show text labels"] == "show_labels"
    assert runtime.support.reference_label_limit_for_annotation_mode("show_labels") == -1
    assert runtime.geometry.reference_hue_options["SFXI score"] == "sfxi_ref__sfxi"
    assert runtime.geometry.reference_hue_kinds["sfxi_ref__sfxi"] == "continuous"
    assert runtime.geometry.reference_hue_options_by_reference_set["reference_sfxi_archive"] == {
        "Black stars": "",
        "SFXI score": "sfxi_ref__sfxi",
        "SFXI logic fidelity": "sfxi_ref__logic_fidelity",
        "SFXI effect scaled": "sfxi_ref__effect_scaled",
    }
    assert runtime.geometry.reference_hue_options_by_reference_set["reference_native_mg1655"] == {"Black stars": ""}
    assert "sfxi_ref__sfxi" in runtime.geometry.reference_required_columns
    assert "sfxi_ref__logic_fidelity" in runtime.geometry.reference_required_columns
    assert "sfxi_ref__effect_scaled" in runtime.geometry.reference_required_columns


def test_reference_annotation_options_keep_label_selection_separate_from_hues() -> None:
    reference_sets = [
        {
            "reference_set_id": "reference_zeta",
            "label": "Zeta reference rows",
            "match_column": "usr_label__primary",
            "label_column": "promoter_standard__display_name",
            "selector_columns": ["promoter_standard__collection_id", "selection_basis"],
        },
        {
            "reference_set_id": "reference_alpha",
            "match_column": "usr_label__primary",
            "label_column": "promoter_standard__display_name",
            "selector_columns": ["promoter_standard__collection_id", "selection_basis"],
        },
        {
            "reference_set_id": "reference_beta",
            "label": "Beta reference rows",
            "match_column": "usr_label__primary",
            "label_column": "usr_label__primary",
            "selector_columns": [],
        },
        {
            "reference_set_id": "reference_beta_duplicate_label",
            "label": "Beta reference rows",
            "match_column": "usr_label__primary",
            "label_column": "usr_label__primary",
            "selector_columns": [],
        },
    ]

    assert _reference_annotation_options(reference_sets) == {
        "Off": "",
        "Zeta reference rows": "reference_zeta",
        "Reference Alpha": "reference_alpha",
        "Beta reference rows": "reference_beta",
        "Beta reference rows (reference_beta_duplicate_label)": "reference_beta_duplicate_label",
    }
    assert _reference_required_columns(reference_sets) == [
        "usr_label__primary",
        "promoter_standard__display_name",
        "promoter_standard__collection_id",
        "selection_basis",
    ]


def test_plot_review_sections_marks_missing_render_path_as_missing(monkeypatch, tmp_path: Path) -> None:
    plot_dir = tmp_path / "outputs" / "plots" / "dataset_overview"
    plot_dir.mkdir(parents=True)
    (plot_dir / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_id": "dataset_overview",
                "status": "ok",
                "stale": False,
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
            question="Fixture question.",
            decision_role="primary",
            encoding="Fixture encoding.",
            scope="Fixture scope.",
            guardrails=["Fixture guardrail."],
            caption="Fixture caption.",
            alt_text="Fixture alt.",
            preprocessing_md="Fixture preprocessing.",
            math_md="Fixture math.",
            rationale_md="Fixture rationale.",
            limitations_md="Fixture limitations.",
            failure_modes_md="Fixture failure modes.",
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

    card = review.sections[0]["cards"][0]
    assert card["render_path"] is None
    assert card["status"] == "missing"


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
    assert review.sections[0]["cards"][0]["alt_text"] == "Validated alt text."
    assert review.sections[0]["cards"][0]["question"] == "Does the plot stay current?"
    assert review.sections[0]["cards"][0]["decision_role"] == "primary"
    assert review.sections[0]["cards"][0]["math_md"] == "Validated math."
    assert review.sections[0]["cards"][0]["failure_modes_md"] == "Validated failure modes."


def test_plot_review_sections_fall_back_to_manifest_status_when_controls_omit_it(monkeypatch, tmp_path: Path) -> None:
    plot_dir = tmp_path / "outputs" / "plots" / "dataset_overview"
    plot_dir.mkdir(parents=True)
    (plot_dir / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_id": "dataset_overview",
                "status": "ok",
                "stale": False,
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
            question="Fixture question.",
            decision_role="primary",
            encoding="Fixture encoding.",
            scope="Fixture scope.",
            guardrails=["Fixture guardrail."],
            caption="Fixture caption.",
            alt_text="Fixture alt.",
            preprocessing_md="Fixture preprocessing.",
            math_md="Fixture math.",
            rationale_md="Fixture rationale.",
            limitations_md="Fixture limitations.",
            failure_modes_md="Fixture failure modes.",
        ),
    )

    review = _plot_review_sections(
        context,
        output_root=tmp_path / "outputs",
        controls={
            "plot_controls": {
                "ordered_plot_ids": ["dataset_overview"],
                "plots": [{"plot_id": "dataset_overview", "deliverable_id": ""}],
            }
        },
    )

    assert review.sections[0]["cards"][0]["render_path"] is None
    assert review.sections[0]["cards"][0]["status"] == "missing"
    assert review.sections[0]["cards"][0]["stale"] is False


def test_plot_review_sections_degrade_one_card_when_manifest_is_invalid(monkeypatch, tmp_path: Path) -> None:
    plot_dir = tmp_path / "outputs" / "plots" / "dataset_overview"
    plot_dir.mkdir(parents=True)
    (plot_dir / "manifest.json").write_text("{invalid json", encoding="utf-8")
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
            question="Fixture question.",
            decision_role="primary",
            encoding="Fixture encoding.",
            scope="Fixture scope.",
            guardrails=["Fixture guardrail."],
            caption="Fixture caption.",
            alt_text="Fixture alt.",
            preprocessing_md="Fixture preprocessing.",
            math_md="Fixture math.",
            rationale_md="Fixture rationale.",
            limitations_md="Fixture limitations.",
            failure_modes_md="Fixture failure modes.",
        ),
    )

    review = _plot_review_sections(
        context,
        output_root=tmp_path / "outputs",
        controls={
            "plot_controls": {
                "ordered_plot_ids": ["dataset_overview"],
                "plots": [{"plot_id": "dataset_overview", "deliverable_id": ""}],
            }
        },
    )

    card = review.sections[0]["cards"][0]
    assert card["status"] == "error"
    assert card["artifact_warning"] is not None
    assert "could not be read" in str(card["artifact_warning"])
    assert card["render_path"] is None


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


def test_plot_review_sections_fail_closed_when_sidecar_resolution_raises(
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
                "semantics": {
                    "plot_id": "dataset_overview",
                    "question": "stale manifest semantics",
                    "decision_role": "primary",
                    "encoding": "stale",
                    "scope": "stale",
                    "guardrails": ["stale"],
                    "caption": "stale",
                    "alt_text": "stale",
                    "preprocessing_md": "stale",
                    "math_md": "stale",
                    "rationale_md": "stale",
                    "limitations_md": "stale",
                    "failure_modes_md": "stale",
                },
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
        lambda context, *, plot_id: (_ for _ in ()).throw(ContractViolationError("broken semantics sidecar")),
    )

    with pytest.raises(ContractViolationError, match="broken semantics sidecar"):
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


def test_plot_review_sections_keep_large_full_population_projection_grid_live_renderable(
    monkeypatch,
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "outputs"
    plot_dir = output_root / "plots" / "appendix_umap_gallery"
    plot_dir.mkdir(parents=True)
    (plot_dir / "gallery.png").write_bytes(b"png")
    (plot_dir / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_id": "appendix_umap_gallery",
                "status": "ok",
                "stale": False,
                "outputs": [{"path": "gallery.png"}],
                "semantics": {"caption": "legacy partial payload"},
            }
        ),
        encoding="utf-8",
    )
    for projection_id in ("proj_a", "proj_b"):
        projection_dir = output_root / "projections" / projection_id
        projection_dir.mkdir(parents=True)
        (projection_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "artifact_id": projection_id,
                    "stats": {
                        "rows": 157_164,
                        "projected_rows": 157_164,
                        "population_rows": 157_164,
                        "is_full_population": True,
                    },
                }
            ),
            encoding="utf-8",
        )

    context = SimpleNamespace(
        config=SimpleNamespace(deliverables={}, plots={"appendix_umap_gallery": object()}),
        require_plot=lambda plot_id: SimpleNamespace(visibility_tier="appendix", semantics_ref="unused"),
        workspace_dir=tmp_path,
    )
    monkeypatch.setattr(
        "dnadesign.latentdna.src.notebooks.browser_runtime._resolve_review_plot_spec",
        lambda context, *, plot_id: SimpleNamespace(
            kind="projection_grid",
            model_dump=lambda mode="json": {
                "plot_id": plot_id,
                "kind": "projection_grid",
                "projection_ids": ["proj_a", "proj_b"],
            },
        ),
    )
    monkeypatch.setattr(
        "dnadesign.latentdna.src.notebooks.browser_runtime.resolve_plot_semantics",
        lambda context, *, plot_id: PlotSemantics(
            plot_id=plot_id,
            question="Fixture question.",
            decision_role="appendix",
            encoding="Fixture encoding.",
            scope="Fixture scope.",
            guardrails=["Fixture guardrail."],
            caption="Fixture caption.",
            alt_text="Fixture alt.",
            preprocessing_md="Fixture preprocessing.",
            math_md="Fixture math.",
            rationale_md="Fixture rationale.",
            limitations_md="Fixture limitations.",
            failure_modes_md="Fixture failure modes.",
        ),
    )

    review = _plot_review_sections(
        context,
        output_root=output_root,
        controls={
            "plot_controls": {
                "ordered_plot_ids": ["appendix_umap_gallery"],
                "plots": [{"plot_id": "appendix_umap_gallery", "deliverable_id": "appendix_umap_gallery"}],
            }
        },
    )

    card = review.sections[0]["cards"][0]
    assert card["live_render"] is True
    assert card["render_mode_note"] is None


def test_plot_review_sections_keep_small_sampled_projection_grid_live_renderable(
    monkeypatch,
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "outputs"
    plot_dir = output_root / "plots" / "appendix_umap_gallery"
    plot_dir.mkdir(parents=True)
    (plot_dir / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_id": "appendix_umap_gallery",
                "status": "ok",
                "stale": False,
                "outputs": [],
                "semantics": {"caption": "legacy partial payload"},
            }
        ),
        encoding="utf-8",
    )
    for projection_id in ("proj_a", "proj_b"):
        projection_dir = output_root / "projections" / projection_id
        projection_dir.mkdir(parents=True)
        (projection_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "artifact_id": projection_id,
                    "stats": {
                        "rows": 2_048,
                        "projected_rows": 2_048,
                        "population_rows": 157_164,
                        "is_full_population": False,
                    },
                }
            ),
            encoding="utf-8",
        )

    context = SimpleNamespace(
        config=SimpleNamespace(deliverables={}, plots={"appendix_umap_gallery": object()}),
        require_plot=lambda plot_id: SimpleNamespace(visibility_tier="appendix", semantics_ref="unused"),
        workspace_dir=tmp_path,
    )
    monkeypatch.setattr(
        "dnadesign.latentdna.src.notebooks.browser_runtime._resolve_review_plot_spec",
        lambda context, *, plot_id: SimpleNamespace(
            kind="projection_grid",
            model_dump=lambda mode="json": {
                "plot_id": plot_id,
                "kind": "projection_grid",
                "projection_ids": ["proj_a", "proj_b"],
            },
        ),
    )
    monkeypatch.setattr(
        "dnadesign.latentdna.src.notebooks.browser_runtime.resolve_plot_semantics",
        lambda context, *, plot_id: PlotSemantics(
            plot_id=plot_id,
            question="Fixture question.",
            decision_role="appendix",
            encoding="Fixture encoding.",
            scope="Fixture scope.",
            guardrails=["Fixture guardrail."],
            caption="Fixture caption.",
            alt_text="Fixture alt.",
            preprocessing_md="Fixture preprocessing.",
            math_md="Fixture math.",
            rationale_md="Fixture rationale.",
            limitations_md="Fixture limitations.",
            failure_modes_md="Fixture failure modes.",
        ),
    )

    review = _plot_review_sections(
        context,
        output_root=output_root,
        controls={
            "plot_controls": {
                "ordered_plot_ids": ["appendix_umap_gallery"],
                "plots": [{"plot_id": "appendix_umap_gallery", "deliverable_id": "appendix_umap_gallery"}],
            }
        },
    )

    card = review.sections[0]["cards"][0]
    assert card["live_render"] is True
    assert card["render_mode_note"] is None
