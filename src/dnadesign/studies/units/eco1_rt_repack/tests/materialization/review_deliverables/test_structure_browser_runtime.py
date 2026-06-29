"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/test_structure_browser_runtime.py

Eco1 interactive structure-browser runtime tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables import (
    materialize_review_deliverables,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables import (
    notebook_structure_browser as structure_browser,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.fixtures import (
    write_deliverable_inputs,
)


def test_structure_browser_runtime_renders_py3dmol_html(tmp_path: Path) -> None:
    write_deliverable_inputs(tmp_path)
    result = materialize_review_deliverables(repo_root=Path.cwd(), output_root=tmp_path, render_chimerax_png=False)

    manifest = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))
    rows = structure_browser.load_structure_browser_rows(
        manifest_root=result.manifest_path.parent,
        deliverables=manifest["deliverables"],
    )
    lookup = structure_browser.structure_browser_lookup(
        rows,
        selected_section="fold_review",
        selected_deliverable_id="interactive_structure_browser_manifest",
    )
    selected = lookup["ProteinMPNN variant rank 1 | WT RMSD 0.82 A | pLDDT 92.4"]

    rendered = structure_browser.render_structure_browser(
        mo=_FakeMo(),
        selected_row=selected,
        structure_ui="<structure-dropdown>",
    )
    rendered_text = str(rendered)

    assert "<iframe" in rendered_text
    assert "3Dmol" in rendered_text
    assert "ec86kit/7V9U reference" in rendered_text
    assert "ProteinMPNN variant rank 1" in rendered_text
    assert "Structure metric summary" in rendered_text
    assert "Mean pLDDT" in rendered_text
    assert "Sequence identity" in rendered_text
    assert "WT-runtime C-alpha RMSD 0.82 A" in rendered_text
    assert "Browser alignment:" in rendered_text
    assert "browser_alignment_status" in rendered_text
    assert "aligned_in_memory_to_reference_ca" in rendered_text
    assert "browser_mapped_ca_rmsd" in rendered_text
    assert "Raw local ColabFold PDB files are not rewritten" in rendered_text


def test_structure_browser_runtime_renders_mask_selection_html(tmp_path: Path) -> None:
    write_deliverable_inputs(tmp_path)
    result = materialize_review_deliverables(repo_root=Path.cwd(), output_root=tmp_path, render_chimerax_png=False)

    manifest = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))
    rows = structure_browser.load_structure_browser_rows(
        manifest_root=result.manifest_path.parent,
        deliverables=manifest["deliverables"],
    )
    lookup = structure_browser.structure_browser_lookup(
        rows,
        selected_section="scaffold_and_mask",
        selected_deliverable_id="mask_structure_browser_manifest",
    )
    selected = lookup["Protected union | 4 residues"]

    rendered = structure_browser.render_structure_browser(
        mo=_FakeMo(),
        selected_row=selected,
        structure_ui="<mask-highlight-dropdown>",
    )
    rendered_text = str(rendered)

    assert "<iframe" in rendered_text
    assert "3Dmol" in rendered_text
    assert "Protected union" in rendered_text
    assert "Reference mask evidence" in rendered_text
    assert "Reference selection:" in rendered_text
    assert "No candidate structure is shown" in rendered_text
    assert (
        "data-selection-id=&quot;protected&quot;" in rendered_text or 'data-selection-id="protected"' in rendered_text
    )

    selection_colors = {
        str(style["color"])
        for row in rows
        if str(row.get("structure_view_mode") or "") == "reference_selection"
        for style in row.get("selection_styles", [])
    }
    assert selection_colors == {"#D55E00"}


class _FakeUi:
    @staticmethod
    def table(rows: list[dict[str, str]], page_size: int) -> dict[str, Any]:
        return {"kind": "table", "rows": rows, "page_size": page_size}


class _FakeMo:
    ui = _FakeUi()

    @staticmethod
    def md(value: str) -> str:
        return value

    @staticmethod
    def Html(value: str) -> str:
        return value

    @staticmethod
    def hstack(items: list[Any], **kwargs: Any) -> dict[str, Any]:
        return {"kind": "hstack", "items": items, "kwargs": kwargs}

    @staticmethod
    def vstack(items: list[Any], **kwargs: Any) -> dict[str, Any]:
        return {"kind": "vstack", "items": items, "kwargs": kwargs}

    @staticmethod
    def accordion(items: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        return {"kind": "accordion", "items": items, "kwargs": kwargs}
