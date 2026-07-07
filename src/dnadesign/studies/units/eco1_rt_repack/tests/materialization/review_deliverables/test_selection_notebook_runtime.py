"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/test_selection_notebook_runtime.py

Eco1 selection-notebook runtime tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables import (
    materialize_review_deliverables,
    notebook_runtime,
    notebook_selection_panel,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.constants import (
    SECTION_FEASIBILITY_AND_HANDOFF,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.visual_inventory import (
    CURRENT_SELECTION_PLOT_IDS,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.fixtures import (
    write_deliverable_inputs,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.runtime_fixtures import (
    FakeMo,
)


def test_selection_panel_table_reads_metrics_from_trace_json(tmp_path: Path) -> None:
    write_deliverable_inputs(tmp_path)
    result = materialize_review_deliverables(repo_root=Path.cwd(), output_root=tmp_path, render_chimerax_png=False)
    manifest = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))
    table_row = next(row for row in manifest["deliverables"] if row["deliverable_id"] == "selection_panel_table")
    rendered = notebook_selection_panel.render_selection_panel_table(
        table_row,
        mo=FakeMo(),
        table_path=result.manifest_path.parent / table_row["path"],
    )
    rows = rendered["items"][1]["rows"]

    assert rows[0]["mutations"] == 2
    assert rows[0]["pLDDT"] == 92.4
    assert rows[0]["WT RMSD A"] == 0.82
    assert rows[0]["cryoEM RMSD A"] == 1.23
    assert rows[0]["unobserved MSA changes"] == 1
    assert rows[0]["near retained DNA/RNA edits"] == 1
    assert rows[0]["near-region charge change"] == 1
    assert rows[0]["near-region chemistry warnings"] == 0
    assert rows[0]["Wang thumb-track edits"] == 0
    assert rows[0]["C-terminal primer-RNA edits"] == 1
    assert len(rendered["items"]) == 2
    assert "Why this row" not in str(rendered)


def test_selection_funnel_and_handoff_readiness_render_from_manifest(tmp_path: Path) -> None:
    write_deliverable_inputs(tmp_path)
    result = materialize_review_deliverables(repo_root=Path.cwd(), output_root=tmp_path, render_chimerax_png=False)
    manifest = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))
    deliverables = {row["deliverable_id"]: row for row in manifest["deliverables"]}

    funnel = notebook_runtime.render_deliverable_artifact(
        deliverables["selection_funnel_summary"],
        mo=FakeMo(),
        manifest_root=result.manifest_path.parent,
    )
    funnel_text = str(funnel)
    assert "Selection policy" in funnel_text
    assert "Accepted candidate pool" in funnel_text
    assert "Conservative-diverse six-row selection" in funnel_text
    assert "design class is context, not a quota" in funnel_text
    assert "hard_gate_status" in funnel_text
    assert "wt_like_not_used_for_selection" in funnel_text
    assert "thread_candidate_alpha" in funnel_text
    assert "not panel-selection evidence" in funnel_text

    readiness = notebook_runtime.render_deliverable_artifact(
        deliverables["selection_handoff_readiness"],
        mo=FakeMo(),
        manifest_root=result.manifest_path.parent,
    )
    readiness_text = str(readiness)
    assert "candidate_handoff.yaml is absent" in readiness_text
    assert "candidate_handoff_sequence_csv_materialized" in readiness_text
    assert "construct_subject_created" in readiness_text
    assert "false" in readiness_text.lower()

    sequences = notebook_runtime.render_deliverable_artifact(
        deliverables["selection_handoff_sequences"],
        mo=FakeMo(),
        manifest_root=result.manifest_path.parent,
    )
    sequence_list_html = sequences["items"][1]
    assert "eco1-handoff-sequence-list" in sequence_list_html
    assert "thread_candidate_alpha" in sequence_list_html
    assert "MKSAGG" in sequence_list_html
    sequence_rows = sequences["items"][2]["rows"]
    assert sequence_rows[0]["candidate_id"] == "thread_candidate_alpha"
    assert sequence_rows[0]["protein_sequence"] == "MKSAGG"
    assert sequence_rows[0]["dna_design_status"] == "not_materialized"


def test_panel_selection_deliverables_follow_review_sequence(tmp_path: Path) -> None:
    write_deliverable_inputs(tmp_path)
    result = materialize_review_deliverables(repo_root=Path.cwd(), output_root=tmp_path, render_chimerax_png=False)
    manifest = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))

    visible_panel_rows = notebook_runtime.section_deliverables(
        notebook_runtime.visual_deliverables(manifest["deliverables"]),
        SECTION_FEASIBILITY_AND_HANDOFF,
    )
    visible_panel_ids = [str(row["deliverable_id"]) for row in visible_panel_rows]
    all_panel_rows = notebook_runtime.section_deliverables(
        manifest["deliverables"],
        SECTION_FEASIBILITY_AND_HANDOFF,
    )
    all_panel_ids = [str(row["deliverable_id"]) for row in all_panel_rows]

    assert visible_panel_ids == [
        *CURRENT_SELECTION_PLOT_IDS,
        "selected_panel_structure_browser_manifest",
    ]
    assert "selection_funnel_summary" not in visible_panel_ids
    assert "selection_panel_table" not in visible_panel_ids
    assert "selection_handoff_sequences" not in visible_panel_ids
    assert all_panel_ids.index("selected_panel_structure_browser_manifest") < all_panel_ids.index(
        "selection_panel_table"
    )
    assert all_panel_ids.index("selection_panel_table") < all_panel_ids.index("selection_handoff_sequences")


def test_residue_frequency_bundle_uses_notebook_owned_selector(tmp_path: Path) -> None:
    write_deliverable_inputs(tmp_path)
    result = materialize_review_deliverables(repo_root=Path.cwd(), output_root=tmp_path, render_chimerax_png=False)
    manifest = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))
    deliverables = {row["deliverable_id"]: row for row in manifest["deliverables"]}
    bundle_row = deliverables["proteinmpnn_residue_frequency_heatmap"]

    view_lookup = notebook_runtime.residue_frequency_view_lookup(bundle_row)
    view_options = list(view_lookup)
    selected_view = notebook_runtime.select_residue_frequency_view(
        selected_label=view_options[-1],
        lookup=view_lookup,
        options=view_options,
    )
    rendered = notebook_runtime.render_residue_frequency_bundle(
        bundle_row,
        mo=FakeMo(),
        manifest_root=result.manifest_path.parent,
        selected_view=selected_view,
        design_class_ui="<fixed-mask-design-class-dropdown>",
    )

    assert notebook_runtime.is_residue_frequency_bundle_deliverable(bundle_row)
    assert len(view_options) == 6
    assert rendered["items"][0] == "<fixed-mask-design-class-dropdown>"
    assert "data:image/svg+xml" in str(rendered)
    assert str(selected_view["label"]) in str(rendered)


def test_non_image_artifact_fallback_uses_manifest_relative_path(tmp_path: Path) -> None:
    tmp_path.joinpath("review_deliverables").mkdir()
    tmp_path.joinpath("review_deliverables", "notes.txt").write_text("fixture\n", encoding="utf-8")

    rendered = notebook_runtime.render_deliverable_artifact(
        {
            "artifact_kind": "text",
            "path": "notes.txt",
            "status": "materialized",
        },
        mo=FakeMo(),
        manifest_root=tmp_path / "review_deliverables",
    )

    assert rendered == "Artifact file: `notes.txt`"
    assert str(tmp_path) not in rendered
