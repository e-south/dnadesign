"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/test_structure_browser_runtime.py

Eco1 candidate structure-browser runtime tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import html as html_lib
from pathlib import Path

import pytest
import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables import (
    materialize_review_deliverables,
    notebook_selection_panel,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables import (
    notebook_structure_browser as structure_browser,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.constants import (
    SECTION_DESIGNS_AND_FOLD_TRIAGE,
    SECTION_FEASIBILITY_AND_HANDOFF,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.fixtures import (
    write_deliverable_inputs,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.runtime_fixtures import (
    FakeMo,
)

from .structure_browser_assertions import (
    assert_candidate_structure_browser_render,
    assert_mutation_overlay_render,
)


def test_structure_browser_runtime_renders_py3dmol_html(tmp_path: Path) -> None:
    write_deliverable_inputs(tmp_path)
    result = materialize_review_deliverables(repo_root=Path.cwd(), output_root=tmp_path, render_chimerax_png=False)

    manifest = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))
    rows = structure_browser.load_structure_browser_rows(
        manifest_root=result.manifest_path.parent,
        deliverables=manifest["deliverables"],
    )
    group_lookup = structure_browser.structure_group_lookup(
        rows,
        selected_section=SECTION_DESIGNS_AND_FOLD_TRIAGE,
        selected_deliverable_id="interactive_structure_browser_manifest",
    )
    assert "0 WT ColabFold baseline" in group_lookup
    assert "1 Passing fold triage (CA RMSD <= 2.0 A; pLDDT >= 90)" in group_lookup
    assert "2 Intermediate fold review band" in group_lookup
    lookup = structure_browser.structure_browser_lookup(
        rows,
        selected_section=SECTION_DESIGNS_AND_FOLD_TRIAGE,
        selected_deliverable_id="interactive_structure_browser_manifest",
        selected_group=group_lookup["1 Passing fold triage (CA RMSD <= 2.0 A; pLDDT >= 90)"],
    )
    selected = lookup["ProteinMPNN variant rank 1 | WT RMSD 0.82 A | pLDDT 92.4"]
    highlight_lookup = structure_browser.structure_highlight_lookup(rows, selected_row=selected)
    selected_highlight = next(row for label, row in highlight_lookup.items() if "SAE F101" in label)

    rendered = structure_browser.render_structure_browser(
        mo=FakeMo(),
        selected_row=selected,
        structure_ui="<structure-dropdown>",
        structure_group_ui="<structure-group-dropdown>",
        structure_highlight_ui="<sae-highlight-dropdown>",
        selected_highlight_row=selected_highlight,
        structure_sidechain_ui="<side-chain-toggle>",
        structure_protein_ui="<protein-color-toggle>",
        structure_dna_ui="<dna-color-toggle>",
        structure_rna_ui="<rna-color-toggle>",
        show_sidechains=True,
    )
    rendered_text = str(rendered)
    unescaped_rendered = html_lib.unescape(rendered_text).replace(" ", "")

    assert_candidate_structure_browser_render(rendered_text, unescaped_rendered)


def test_structure_browser_runtime_can_toggle_reference_and_mutation_overlay(tmp_path: Path) -> None:
    write_deliverable_inputs(tmp_path)
    result = materialize_review_deliverables(repo_root=Path.cwd(), output_root=tmp_path, render_chimerax_png=False)

    manifest = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))
    rows = structure_browser.load_structure_browser_rows(
        manifest_root=result.manifest_path.parent,
        deliverables=manifest["deliverables"],
    )
    lookup = structure_browser.structure_browser_lookup(
        rows,
        selected_section=SECTION_DESIGNS_AND_FOLD_TRIAGE,
        selected_deliverable_id="interactive_structure_browser_manifest",
        selected_group="1 Passing fold triage (CA RMSD <= 2.0 A; pLDDT >= 90)",
    )
    selected = lookup["ProteinMPNN variant rank 1 | WT RMSD 0.82 A | pLDDT 92.4"]

    rendered = structure_browser.render_structure_browser(
        mo=FakeMo(),
        selected_row=selected,
        structure_ui="<structure-dropdown>",
        structure_group_ui="<structure-group-dropdown>",
        structure_background_ui="<reference-background-toggle>",
        structure_mutation_ui="<mutation-toggle>",
        structure_sidechain_ui="<side-chain-toggle>",
        show_reference_background=False,
        show_mutation_differences=True,
        show_sidechains=False,
    )
    rendered_text = str(rendered)
    unescaped_rendered = html_lib.unescape(rendered_text).replace(" ", "")

    assert_mutation_overlay_render(rendered_text, unescaped_rendered)


def test_selected_panel_structure_browser_uses_expanded_selection_rows(tmp_path: Path) -> None:
    write_deliverable_inputs(tmp_path)
    result = materialize_review_deliverables(repo_root=Path.cwd(), output_root=tmp_path, render_chimerax_png=False)

    manifest = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))
    rows = structure_browser.load_structure_browser_rows(
        manifest_root=result.manifest_path.parent,
        deliverables=manifest["deliverables"],
    )
    selected_rows = [row for row in rows if row.get("_deliverable_id") == "selected_panel_structure_browser_manifest"]
    assert {row["candidate_id"] for row in selected_rows} == {
        "wild_type",
        "thread_candidate_alpha",
        "thread_candidate_beta",
    }
    group_lookup = structure_browser.structure_group_lookup(
        selected_rows,
        selected_section=SECTION_FEASIBILITY_AND_HANDOFF,
        selected_deliverable_id="selected_panel_structure_browser_manifest",
    )
    assert "1 Selected panel: clade9_p25_contact5a" in group_lookup
    lookup = structure_browser.structure_browser_lookup(
        selected_rows,
        selected_section=SECTION_FEASIBILITY_AND_HANDOFF,
        selected_deliverable_id="selected_panel_structure_browser_manifest",
        selected_group=group_lookup["1 Selected panel: clade9_p25_contact5a"],
    )
    selected = lookup["ProteinMPNN variant rank 1 | WT RMSD 0.82 A | pLDDT 92.4"]

    rendered = structure_browser.render_structure_browser(
        mo=FakeMo(),
        selected_row=selected,
        structure_ui="<structure-dropdown>",
        structure_group_ui="<structure-group-dropdown>",
        show_sidechains=True,
    )
    rendered_text = str(rendered)

    assert "Variant dashboard" in rendered_text
    assert "Selection slot" in rendered_text
    assert "clade9_p25_contact5a" in rendered_text
    assert "MSA observed fraction" in rendered_text
    assert "NA-facing charge change" in rendered_text
    assert "Distal scaffold changes" in rendered_text


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
    assert rows[0]["NA-facing charge change"] == 1


def test_structure_browser_manifest_rejects_missing_declared_pdb(tmp_path: Path) -> None:
    write_deliverable_inputs(tmp_path)
    full_structure_set_path = tmp_path / "foldcheck_review" / "foldcheck_full_structure_set.yaml"
    payload = yaml.safe_load(full_structure_set_path.read_text(encoding="utf-8"))
    payload["structures"][0]["local_model_artifact_path"] = "structures/full_fold_set/missing_model.pdb"
    full_structure_set_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="declared structure path is missing"):
        materialize_review_deliverables(repo_root=Path.cwd(), output_root=tmp_path, render_chimerax_png=False)
