"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/test_structure_browser_highlights.py

Eco1 candidate structure-browser highlight-overlay tests.

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
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables import (
    structure_browser_common as browser_colors,
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


def test_candidate_structure_browser_can_overlay_reference_mask_highlights(tmp_path: Path) -> None:
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
    highlight_lookup = structure_browser.structure_highlight_lookup(rows, selected_row=selected)

    assert "No residue highlight" in highlight_lookup
    selected_highlight = highlight_lookup["Fixed mask: Clade 9 p25 + 5 A | 4 residues"]
    rendered = structure_browser.render_structure_browser(
        mo=FakeMo(),
        selected_row=selected,
        structure_ui="<structure-dropdown>",
        structure_highlight_ui="<residue-highlight-dropdown>",
        selected_highlight_row=selected_highlight,
        structure_sidechain_ui="<side-chain-toggle>",
    )
    rendered_text = str(rendered)
    unescaped_rendered = html_lib.unescape(rendered_text).replace(" ", "")

    assert "<residue-highlight-dropdown>" in rendered_text
    assert "Selected residue highlight" in rendered_text
    assert "Fixed mask: Clade 9 p25 + 5 A" in rendered_text
    assert "Selected SAE feature" not in rendered_text
    assert f'"stick":{{"color":"{browser_colors.RESIDUE_CATEGORY_HIGHLIGHT_COLOR}","radius":0.22}}' in (
        unescaped_rendered
    )
