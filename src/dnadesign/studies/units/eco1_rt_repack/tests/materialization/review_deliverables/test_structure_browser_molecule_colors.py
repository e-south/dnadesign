"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/test_structure_browser_molecule_colors.py

Eco1 structure-browser molecule-color tests.

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


def test_structure_browser_runtime_can_toggle_molecule_class_colors(tmp_path: Path) -> None:
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
        structure_protein_ui="<protein-color-toggle>",
        structure_dna_ui="<dna-color-toggle>",
        structure_rna_ui="<rna-color-toggle>",
        highlight_protein=True,
        highlight_dna=True,
        highlight_rna=True,
    )
    rendered_text = str(rendered)
    unescaped_rendered = html_lib.unescape(rendered_text).replace(" ", "")

    assert "<protein-color-toggle>" in rendered_text
    assert "<dna-color-toggle>" in rendered_text
    assert "<rna-color-toggle>" in rendered_text
    assert "Protein" in rendered_text
    assert "DNA" in rendered_text
    assert "RNA" in rendered_text
    assert '"resn":["DA","DC","DG","DT"]' in unescaped_rendered
    assert '"resn":["A","C","G","I","U"]' in unescaped_rendered
