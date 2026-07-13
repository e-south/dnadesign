"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/test_structure_browser_runtime.py

Eco1 candidate structure-browser rendering tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import html as html_lib
from pathlib import Path

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables import (
    materialize_review_deliverables,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables import (
    notebook_structure_browser as structure_browser,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.constants import (
    SECTION_DESIGNS_AND_FOLD_TRIAGE,
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


def _candidate_context(tmp_path: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
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
    assert set(group_lookup) >= {
        "0 WT ColabFold baseline",
        "1 Passing fold triage (CA RMSD <= 2.0 A; pLDDT >= 90)",
        "2 Intermediate fold review band",
    }
    lookup = structure_browser.structure_browser_lookup(
        rows,
        selected_section=SECTION_DESIGNS_AND_FOLD_TRIAGE,
        selected_deliverable_id="interactive_structure_browser_manifest",
        selected_group=group_lookup["1 Passing fold triage (CA RMSD <= 2.0 A; pLDDT >= 90)"],
    )
    return lookup["ProteinMPNN variant rank 1 | WT RMSD 0.82 A | pLDDT 92.4"], rows


def test_structure_browser_runtime_renders_py3dmol_html(tmp_path: Path) -> None:
    selected, rows = _candidate_context(tmp_path)
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
        structure_dna_visible_ui="<show-dna-toggle>",
        structure_rna_visible_ui="<show-rna-toggle>",
        show_sidechains=True,
        highlight_dna=True,
        highlight_rna=True,
    )
    rendered_text = str(rendered)
    assert_candidate_structure_browser_render(rendered_text, html_lib.unescape(rendered_text).replace(" ", ""))


def test_structure_browser_runtime_can_toggle_reference_and_mutation_overlay(tmp_path: Path) -> None:
    selected, _rows = _candidate_context(tmp_path)
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
    assert_mutation_overlay_render(rendered_text, html_lib.unescape(rendered_text).replace(" ", ""))
    assert "Reference nucleic acids" in rendered_text


def test_candidate_highlight_rebinds_reference_selection_to_query_model(tmp_path: Path) -> None:
    selected, rows = _candidate_context(tmp_path)
    highlight_lookup = structure_browser.structure_highlight_lookup(rows, selected_row=selected)
    protected = next(row for label, row in highlight_lookup.items() if label.startswith("Protected residues"))

    rendered = structure_browser.render_structure_browser(
        mo=FakeMo(),
        selected_row=selected,
        structure_ui=None,
        selected_highlight_row=protected,
        show_reference_background=False,
        show_sidechains=True,
    )

    rendered_text = str(rendered)
    assert "Interactive structure viewer failed to render" not in rendered_text
    assert str(selected["candidate_id"]) in rendered_text
