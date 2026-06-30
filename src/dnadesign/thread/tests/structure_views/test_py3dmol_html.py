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
    StructureViewMoleculeStyle,
    StructureViewSelectionStyle,
    StructureViewSpec,
    render_structure_view_html,
    structure_view_backend_available,
    summarize_pdb_atom_content,
    summarize_structure_atom_content,
)

_MINIMAL_PDB = """\
ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00 80.00           N
ATOM      2  CA  GLY A   1       1.458   0.000   0.000  1.00 80.00           C
ATOM      3  C   GLY A   1       2.000   1.400   0.000  1.00 80.00           C
ATOM      4  O   GLY A   1       1.300   2.300   0.000  1.00 80.00           O
END
"""

_SIDECHAIN_PDB = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 80.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 80.00           C
ATOM      3  CB  ALA A   1       1.800  -1.200   0.000  1.00 80.00           C
ATOM      4  C   ALA A   1       2.000   1.400   0.000  1.00 80.00           C
ATOM      5  O   ALA A   1       1.300   2.300   0.000  1.00 80.00           O
END
"""

_PROTEIN_AND_NUCLEIC_PDB = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 80.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 80.00           C
ATOM      3  CB  ALA A   1       1.800  -1.200   0.000  1.00 80.00           C
ATOM      4  C   ALA A   1       2.000   1.400   0.000  1.00 80.00           C
ATOM      5  O   ALA A   1       1.300   2.300   0.000  1.00 80.00           O
HETATM    6  P    DA B   1       4.000   0.000   0.000  1.00 80.00           P
HETATM    7  O3'  DA B   1       4.500   0.500   0.000  1.00 80.00           O
HETATM    8  P     U C   2       5.000   0.000   0.000  1.00 80.00           P
HETATM    9  O3'   U C   2       5.500   0.500   0.000  1.00 80.00           O
END
"""

_SIDECHAIN_MMCIF = """\
data_fixture
ATOM 1 N N . SER A 1 3 0.000 0.000 0.000 A 3 ? 1.00 80.00 1
ATOM 2 C CA . SER A 1 3 1.000 0.000 0.000 A 3 ? 1.00 80.00 1
ATOM 3 C CB . SER A 1 3 1.000 1.000 0.000 A 3 ? 1.00 80.00 1
ATOM 4 C C . SER A 1 3 2.000 0.000 0.000 A 3 ? 1.00 80.00 1
ATOM 5 O O . SER A 1 3 2.500 0.500 0.000 A 3 ? 1.00 80.00 1
HETATM 6 P P . DA D 2 1 4.000 0.000 0.000 D 1 ? 1.00 80.00 1
#
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
    assert "width:100%; max-width:100%" in html
    assert "height:500px" in html
    assert "height:512px" not in html
    assert "margin:0 auto" in html
    assert "text-align:center" in html
    assert "justify-content:center" in html
    assert "setProjection(&quot;orthographic&quot;)" in html
    assert "setViewStyle({&quot;style&quot;: &quot;outline&quot;})" in html
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_py3dmol_backend_keeps_description_metadata_nonvisual() -> None:
    html = render_structure_view_html(
        StructureViewSpec(
            title="Reference mask context",
            description="Shows the reference backbone with one declared residue set highlighted.",
            interpretation_limit="This is a review view, not fold validation or activity evidence.",
            models=(StructureViewModel("reference", _MINIMAL_PDB, label="Reference", color="#f2efe8"),),
        )
    )

    assert "What this structure view shows" not in html
    assert "Shows the reference backbone with one declared residue set highlighted." in html
    assert "Interpretation limit:" not in html
    assert "not fold validation or activity evidence" in html
    assert "structure-view-sr-only" in html
    assert "aria-describedby=" in html


def test_py3dmol_backend_can_show_sidechains_and_persist_camera() -> None:
    html = render_structure_view_html(
        StructureViewSpec(
            title="Predicted structure review",
            models=(
                StructureViewModel(
                    "query",
                    _MINIMAL_PDB,
                    label="Query",
                    color="#0072B2",
                    show_sidechains=True,
                ),
            ),
            camera_memory_key="eco1-review:test-structure-browser",
        )
    )

    unescaped_html = html_lib.unescape(html).replace(" ", "")
    assert "localStorage" in html
    assert "eco1-review:test-structure-browser" in html
    assert "zoom(1.35)" in html
    assert "translate(0,0)" in html
    assert "twoFingerPan" in html
    assert "event.preventDefault()" in html
    assert "event.stopPropagation()" in html
    assert "registerPanTarget(container)" in html
    assert "registerPanTarget(canvas)" in html
    assert "registerPanTarget(document)" in html
    assert "{passive: false, capture: true}" in html
    assert "viewer.translateScene.bind(viewer)" in html
    assert "const pan = translateScene || translate" in html
    assert "pan(-event.deltaX * panScale, -event.deltaY * panScale)" in html
    assert '"not":{"atom":["N","C","O","OXT"]}' in unescaped_html
    expected_residue_selection = (
        '"resn":["ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",'
        '"LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"]'
    )
    assert expected_residue_selection in unescaped_html
    assert '"stick":{"color":"#0072B2","radius":0.16}' in unescaped_html


def test_py3dmol_backend_scopes_residue_highlights_to_protein_residues() -> None:
    html = render_structure_view_html(
        StructureViewSpec(
            title="Protein-scoped selection",
            models=(StructureViewModel("reference", _PROTEIN_AND_NUCLEIC_PDB, label="Reference"),),
            selection_styles=(
                StructureViewSelectionStyle(
                    selection_id="protein_only",
                    model_id="reference",
                    label="Protein residue 1",
                    residue_numbers=(1,),
                    color="#D55E00",
                ),
            ),
        )
    )

    unescaped_html = html_lib.unescape(html).replace(" ", "")
    assert '"selection_id":"protein_only"' not in unescaped_html
    assert '"resi":[1]' in unescaped_html
    assert (
        '"resn":["ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",'
        '"LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"]'
    ) in unescaped_html
    assert '"resn":["DA","DC","DG","DT"]' not in unescaped_html


def test_py3dmol_backend_can_color_molecule_classes_independently() -> None:
    html = render_structure_view_html(
        StructureViewSpec(
            title="Molecule-class coloring",
            models=(StructureViewModel("reference", _PROTEIN_AND_NUCLEIC_PDB, label="Reference"),),
            molecule_styles=(
                StructureViewMoleculeStyle("protein", "reference", label="Protein", color="#0072B2"),
                StructureViewMoleculeStyle("dna", "reference", label="DNA", color="#E69F00"),
                StructureViewMoleculeStyle("rna", "reference", label="RNA", color="#009E73"),
            ),
        )
    )

    unescaped_html = html_lib.unescape(html).replace(" ", "")
    assert "Protein" in html
    assert "DNA" in html
    assert "RNA" in html
    assert '"resn":["DA","DC","DG","DT"]' in unescaped_html
    assert '"resn":["A","C","G","I","U"]' in unescaped_html
    assert '"resn":["ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",' in unescaped_html
    assert '"stick":{"color":"#E69F00","radius":0.18}' in unescaped_html
    assert '"stick":{"color":"#009E73","radius":0.18}' in unescaped_html


def test_py3dmol_backend_maps_mmcif_contract_to_3dmol_cif_loader() -> None:
    html = render_structure_view_html(
        StructureViewSpec(
            title="CIF reference review",
            models=(
                StructureViewModel(
                    "reference",
                    _SIDECHAIN_MMCIF,
                    structure_format="mmcif",
                    label="Reference",
                    color="#d6d6d6",
                    show_sidechains=True,
                ),
            ),
        )
    )

    unescaped_html = html_lib.unescape(html).replace(" ", "")
    assert 'addModel("data_fixture\\nATOM' in unescaped_html
    assert '","cif");' in unescaped_html
    assert '","mmcif");' not in unescaped_html


def test_structure_atom_content_summary_detects_sidechain_atoms() -> None:
    backbone_content = summarize_pdb_atom_content(_MINIMAL_PDB)
    sidechain_content = summarize_pdb_atom_content(_SIDECHAIN_PDB)

    assert backbone_content.atom_count == 4
    assert backbone_content.residue_count == 1
    assert backbone_content.sidechain_atom_count == 0
    assert not backbone_content.has_sidechain_atoms
    assert backbone_content.scope_label == "backbone_only_or_no_sidechain_atoms"
    assert sidechain_content.atom_count == 5
    assert sidechain_content.residue_count == 1
    assert sidechain_content.sidechain_atom_count == 1
    assert sidechain_content.sidechain_residue_count == 1
    assert sidechain_content.has_sidechain_atoms
    assert sidechain_content.scope_label == "sidechain_atoms_present"


def test_structure_atom_content_summary_detects_mmcif_protein_sidechains_only() -> None:
    content = summarize_structure_atom_content(_SIDECHAIN_MMCIF, structure_format="mmcif")

    assert content.atom_count == 5
    assert content.residue_count == 1
    assert content.sidechain_atom_count == 1
    assert content.sidechain_residue_count == 1
    assert content.has_sidechain_atoms


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
    assert '"resn":["ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",' in unescaped_html
    assert "#D55E00" in html
