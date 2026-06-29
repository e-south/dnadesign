"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/tests/structure_views/test_py3dmol_html.py

Tests for generic py3Dmol-backed structure views.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import html as html_lib

import pytest

from dnadesign.thread.structure_views import (
    StructureViewModel,
    StructureViewSelectionStyle,
    StructureViewSpec,
    render_structure_view_html,
    structure_view_backend_available,
)

_MINIMAL_PDB = """\
ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00 80.00           N
ATOM      2  CA  GLY A   1       1.458   0.000   0.000  1.00 80.00           C
ATOM      3  C   GLY A   1       2.000   1.400   0.000  1.00 80.00           C
ATOM      4  O   GLY A   1       1.300   2.300   0.000  1.00 80.00           O
END
"""


def test_py3dmol_backend_renders_interactive_html() -> None:
    assert structure_view_backend_available("py3dmol")
    html = render_structure_view_html(
        StructureViewSpec(
            title="Reference and query structure",
            subtitle="Aligned to a shared reference pose",
            models=(
                StructureViewModel("reference", _MINIMAL_PDB, label="Reference", color="#d6d6d6"),
                StructureViewModel("query", _MINIMAL_PDB, label="Query", color="#0072B2"),
            ),
        )
    )

    assert "3Dmol" in html
    assert "Reference and query structure" in html
    assert "Aligned to a shared reference pose" in html
    assert "Reference" in html
    assert "Query" in html
    assert "<iframe" in html
    assert "srcdoc=" in html
    assert "sandbox=" in html
    assert "width:min(100%, 732px)" in html
    assert "margin:0 auto" in html
    assert "text-align:center" in html
    assert "justify-content:center" in html
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_structure_view_contract_rejects_unsupported_backend() -> None:
    with pytest.raises(ValueError, match="Unsupported structure-view backend"):
        render_structure_view_html(
            StructureViewSpec(title="x", models=(StructureViewModel("m", _MINIMAL_PDB),)),
            backend="missing",
        )


def test_structure_view_contract_rejects_empty_structure() -> None:
    with pytest.raises(ValueError, match="structure_text is required"):
        StructureViewSpec(title="x", models=(StructureViewModel("m", ""),)).validate()


def test_py3dmol_backend_renders_residue_selection_styles() -> None:
    html = render_structure_view_html(
        StructureViewSpec(
            title="Reference mask context",
            models=(StructureViewModel("reference", _MINIMAL_PDB, label="Reference", color="#f2efe8"),),
            selection_styles=(
                StructureViewSelectionStyle(
                    selection_id="active_site",
                    model_id="reference",
                    label="Active-site residues",
                    residue_numbers=(1,),
                    color="#D55E00",
                ),
            ),
        )
    )

    unescaped_html = html_lib.unescape(html).replace(" ", "")
    assert "Active-site residues" in html
    assert "active_site" in html
    assert '"resi":[1]' in unescaped_html
    assert "#D55E00" in html
