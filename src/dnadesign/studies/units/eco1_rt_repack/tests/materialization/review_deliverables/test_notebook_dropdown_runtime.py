"""No-dead-end contracts for every Eco1 review-notebook dropdown."""

from __future__ import annotations

from pathlib import Path

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables import (
    materialize_review_deliverables,
    notebook_runtime,
    notebook_sae_features,
    notebook_structure_browser,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.fixtures import (
    write_deliverable_inputs,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.runtime_fixtures import (
    FakeMo,
)


def test_notebook_dropdown_rows_resolve_to_existing_artifacts(tmp_path: Path) -> None:
    write_deliverable_inputs(tmp_path)
    result = materialize_review_deliverables(repo_root=Path.cwd(), output_root=tmp_path, render_chimerax_png=False)
    _, deliverables, _, _ = notebook_runtime.load_review_manifest(str(result.notebook_path))

    for lane_id in notebook_runtime.review_lane_lookup(deliverables).values():
        visible_rows = notebook_runtime.visual_deliverables(deliverables, selected_lane=lane_id)
        evidence_rows = notebook_runtime.evidence_deliverables(deliverables, selected_lane=lane_id)
        for row in [*visible_rows, *evidence_rows]:
            assert row["_notebook_artifact_exists"], row["deliverable_id"]


def test_every_notebook_dropdown_choice_renders_without_a_dead_end(tmp_path: Path) -> None:
    write_deliverable_inputs(tmp_path)
    result = materialize_review_deliverables(repo_root=Path.cwd(), output_root=tmp_path, render_chimerax_png=False)
    _, deliverables, _, manifest_root = notebook_runtime.load_review_manifest(str(result.notebook_path))
    all_structure_rows = notebook_structure_browser.load_structure_browser_rows(
        manifest_root=manifest_root,
        deliverables=deliverables,
    )

    for lane_id in notebook_runtime.review_lane_lookup(deliverables).values():
        for row in notebook_runtime.visual_deliverables(deliverables, selected_lane=lane_id):
            deliverable_id = str(row["deliverable_id"])
            if notebook_runtime.is_interactive_structure_deliverable(row):
                structure_rows = [
                    row for row in all_structure_rows if str(row.get("_deliverable_id") or "") == deliverable_id
                ]
                assert structure_rows, deliverable_id
                for selected_row in structure_rows:
                    highlight_rows = list(
                        notebook_structure_browser.structure_highlight_lookup(
                            all_structure_rows,
                            selected_row=selected_row,
                        ).values()
                    ) or [None]
                    for selected_highlight_row in highlight_rows:
                        rendered = notebook_structure_browser.render_structure_browser(
                            mo=FakeMo(),
                            selected_row=selected_row,
                            selected_highlight_row=selected_highlight_row,
                            structure_ui="structure selector",
                            show_reference_background=True,
                            show_sidechains=True,
                            show_protein_surface=False,
                            show_dna=True,
                            show_rna=True,
                        )
                        _assert_no_render_dead_end(rendered, deliverable_id=deliverable_id)
                    for display_state in _STRUCTURE_DISPLAY_STATES:
                        rendered = notebook_structure_browser.render_structure_browser(
                            mo=FakeMo(),
                            selected_row=selected_row,
                            selected_highlight_row=highlight_rows[0],
                            structure_ui="structure selector",
                            **display_state,
                        )
                        _assert_no_render_dead_end(rendered, deliverable_id=deliverable_id)
                continue
            if notebook_runtime.is_policy_residue_frequency_deliverable(row):
                policy_views = notebook_runtime.policy_frequency_view_lookup(row)
                assert policy_views, deliverable_id
                for selected_view in policy_views.values():
                    rendered = notebook_runtime.render_policy_frequency_bundle(
                        row,
                        mo=FakeMo(),
                        manifest_root=manifest_root,
                        selected_view=selected_view,
                    )
                    _assert_no_render_dead_end(rendered, deliverable_id=deliverable_id)
                continue
            if notebook_sae_features.is_sae_feature_heatmap_deliverable(row):
                payload = notebook_sae_features.load_sae_feature_heatmap_manifest(
                    manifest_root=manifest_root,
                    selected_visual=row,
                )
                feature_lookup = notebook_sae_features.sae_heatmap_feature_lookup(payload)
                assert feature_lookup, deliverable_id
                for feature_index in feature_lookup.values():
                    rendered = notebook_sae_features.render_sae_feature_heatmap(
                        mo=FakeMo(),
                        heatmap_manifest=payload,
                        selected_feature_index=feature_index,
                        feature_ui=None,
                    )
                    _assert_no_render_dead_end(rendered, deliverable_id=deliverable_id)
                continue
            rendered = notebook_runtime.render_deliverable_panel(row, mo=FakeMo(), manifest_root=manifest_root)
            _assert_no_render_dead_end(rendered, deliverable_id=deliverable_id)


def _assert_no_render_dead_end(rendered: object, *, deliverable_id: str) -> None:
    rendered_text = str(rendered)
    dead_end_messages = (
        "Artifact unavailable",
        "failed to render",
        "is skipped because",
        "is unavailable",
        "is missing",
    )
    assert not any(message in rendered_text for message in dead_end_messages), deliverable_id


_STRUCTURE_DISPLAY_STATES = (
    {
        "show_reference_background": True,
        "show_mutation_differences": False,
        "show_sidechains": True,
        "show_protein_surface": False,
        "show_dna": True,
        "show_rna": True,
    },
    {
        "show_reference_background": False,
        "show_mutation_differences": True,
        "show_sidechains": False,
        "show_protein_surface": False,
        "show_dna": True,
        "show_rna": True,
    },
    {
        "show_reference_background": True,
        "show_mutation_differences": False,
        "show_sidechains": True,
        "show_protein_surface": True,
        "show_dna": True,
        "show_rna": True,
    },
    {
        "show_reference_background": True,
        "show_mutation_differences": False,
        "show_sidechains": True,
        "show_protein_surface": False,
        "show_dna": False,
        "show_rna": True,
    },
    {
        "show_reference_background": True,
        "show_mutation_differences": False,
        "show_sidechains": True,
        "show_protein_surface": False,
        "show_dna": True,
        "show_rna": False,
    },
)


def test_missing_file_rows_are_not_dropdown_options(tmp_path: Path) -> None:
    write_deliverable_inputs(tmp_path)
    result = materialize_review_deliverables(repo_root=Path.cwd(), output_root=tmp_path, render_chimerax_png=False)
    manifest = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))
    stale_row = next(
        row for row in manifest["deliverables"] if row["deliverable_id"] == "selection_hypothesis_panel_flow"
    )
    (result.manifest_path.parent / stale_row["path"]).unlink()

    _, deliverables, _, _ = notebook_runtime.load_review_manifest(str(result.notebook_path))
    visible_ids = {
        str(row["deliverable_id"])
        for row in notebook_runtime.visual_deliverables(deliverables, selected_lane="main_review")
    }
    assert "selection_hypothesis_panel_flow" not in visible_ids


def test_non_image_artifact_fallback_uses_manifest_relative_path(tmp_path: Path) -> None:
    tmp_path.joinpath("review_deliverables").mkdir()
    tmp_path.joinpath("review_deliverables", "notes.txt").write_text("fixture\n", encoding="utf-8")

    rendered = notebook_runtime.render_deliverable_artifact(
        {"artifact_kind": "text", "path": "notes.txt", "status": "materialized"},
        mo=FakeMo(),
        manifest_root=tmp_path / "review_deliverables",
    )

    assert rendered == "Artifact file: `notes.txt`"
    assert str(tmp_path) not in rendered
