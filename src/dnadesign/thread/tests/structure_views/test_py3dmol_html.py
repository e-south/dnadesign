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
    filter_structure_text_by_molecule_classes,
    molecule_classes_in_structure_text,
    render_structure_view_html,
    structure_view_backend_available,
    summarize_pdb_atom_content,
    summarize_structure_atom_content,
)
from dnadesign.thread.structure_views._mmcif import (
    _browser_cif_token,
    _quote_3dmol_atom_name,
    serialize_mmcif_atom_sites_for_3dmol,
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
HETATM    7  C4'  DA B   1       4.400   0.400   0.000  1.00 80.00           C
HETATM    8  N1   DA B   1       4.900   0.900   0.000  1.00 80.00           N
HETATM    9  P    DA B   2       5.000   0.000   0.000  1.00 80.00           P
HETATM   10  C4'  DA B   2       5.400   0.400   0.000  1.00 80.00           C
HETATM   11  N1   DA B   2       5.900   0.900   0.000  1.00 80.00           N
HETATM   12  P    DA B   3       6.000   0.000   0.000  1.00 80.00           P
HETATM   13  C4'  DA B   3       6.400   0.400   0.000  1.00 80.00           C
HETATM   14  N1   DA B   3       6.900   0.900   0.000  1.00 80.00           N
HETATM   15  P     U C   2       5.000   0.000   1.000  1.00 80.00           P
HETATM   16  C4'   U C   2       5.400   0.400   1.000  1.00 80.00           C
HETATM   17  N1    U C   2       5.900   0.900   1.000  1.00 80.00           N
HETATM   18  P     U C   3       6.000   0.000   1.000  1.00 80.00           P
HETATM   19  C4'   U C   3       6.400   0.400   1.000  1.00 80.00           C
HETATM   20  N1    U C   3       6.900   0.900   1.000  1.00 80.00           N
HETATM   21  P     U C   4       7.000   0.000   1.000  1.00 80.00           P
HETATM   22  C4'   U C   4       7.400   0.400   1.000  1.00 80.00           C
HETATM   23  N1    U C   4       7.900   0.900   1.000  1.00 80.00           N
END
"""

_SIDECHAIN_MMCIF = """\
data_fixture
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.auth_asym_id
_atom_site.auth_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.pdbx_PDB_model_num
ATOM 1 N N . SER A 1 3 0.000 0.000 0.000 A 3 ? 1.00 80.00 1
ATOM 2 C CA . SER A 1 3 1.000 0.000 0.000 A 3 ? 1.00 80.00 1
ATOM 3 C CB . SER A 1 3 1.000 1.000 0.000 A 3 ? 1.00 80.00 1
ATOM 4 C C . SER A 1 3 2.000 0.000 0.000 A 3 ? 1.00 80.00 1
ATOM 5 O O . SER A 1 3 2.500 0.500 0.000 A 3 ? 1.00 80.00 1
HETATM 6 P P . DA D 2 1 4.000 0.000 0.000 D 1 ? 1.00 80.00 1
#
"""

_ATOM_NAME_FALLBACK_MMCIF = """\
data_atom_name_fallback
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.auth_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.auth_asym_id
_atom_site.auth_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.pdbx_PDB_model_num
ATOM 1 C . CA . SER A 1 3 1.000 1.000 0.000 A 3 ? 1.00 80.00 1
ATOM 2 C CB AUTH . SER A 1 3 2.000 1.000 0.000 A 3 ? 1.00 80.00 1
#
"""

_MIXED_POLYMER_MMCIF_WITH_UNQUOTED_PRIME_ATOMS = """\
data_mixed_polymer
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.auth_asym_id
_atom_site.auth_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.pdbx_PDB_model_num
ATOM 1 N N . GLY A 1 1 0.000 0.000 0.000 A 1 ? 1.00 80.00 1
ATOM 2 C CA . GLY A 1 1 1.458 0.000 0.000 A 1 ? 1.00 80.00 1
ATOM 3 C C . GLY A 1 1 2.000 1.400 0.000 A 1 ? 1.00 80.00 1
ATOM 4 O O . GLY A 1 1 1.300 2.300 0.000 A 1 ? 1.00 80.00 1
ATOM 5 P P . DG B 2 1 4.000 0.000 0.000 D 1 ? 1.00 80.00 1
ATOM 6 O O5' . DG B 2 1 4.200 0.200 0.000 D 1 ? 1.00 80.00 1
ATOM 7 C C4' . DG B 2 1 4.400 0.400 0.000 D 1 ? 1.00 80.00 1
ATOM 8 N N9 . DG B 2 1 4.900 0.900 0.000 D 1 ? 1.00 80.00 1
#
"""

_DNA_ONLY_MMCIF = "\n".join(line for line in _SIDECHAIN_MMCIF.splitlines() if not line.startswith("ATOM "))
_PROTEIN_ONLY_MMCIF = "\n".join(line for line in _SIDECHAIN_MMCIF.splitlines() if not line.startswith("HETATM "))
_LIGAND_ONLY_MMCIF = _DNA_ONLY_MMCIF.replace(" DA D 2 1 ", " HEM L 2 1 ").replace(
    " D 1 ? 1.00 80.00 1",
    " L 1 ? 1.00 80.00 1",
)
_DNA_ONLY_MMCIF_WITH_ENTITY_CATEGORY = _DNA_ONLY_MMCIF.replace(
    "loop_\n_atom_site.group_PDB",
    "loop_\n_entity.id\n_entity.type\n1 polymer\n#\nloop_\n_atom_site.group_PDB",
)
_DNA_ONLY_MULTIMODEL_MMCIF = _DNA_ONLY_MMCIF.replace(
    "HETATM 6 P P . DA D 2 1 4.000 0.000 0.000 D 1 ? 1.00 80.00 1",
    "HETATM 6 P P . DA D 2 1 4.000 0.000 0.000 D 1 ? 1.00 80.00 1\n"
    "HETATM 7 P P . DA D 2 1 5.000 0.000 0.000 D 1 ? 1.00 80.00 2",
)
_EMPTY_ATOM_SITE_MMCIF = "\n".join(
    line for line in _SIDECHAIN_MMCIF.splitlines() if not line.startswith(("ATOM ", "HETATM "))
)

_DNA_RESIDUE_SELECTION = '"resn":["DA","DC","DG","DT"]'
_RNA_RESIDUE_SELECTION = '"resn":["A","C","G","I","U"]'


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
    assert 'sandbox="allow-scripts"' in html
    assert "allow-same-origin" not in html
    assert "width:100%; max-width:100%" in html
    assert "height:500px" in html
    assert "height:512px" not in html
    assert "margin:0 auto" in html
    assert "text-align:center" in html
    assert "justify-content:center" in html
    assert "setBackgroundColor(&quot;white&quot;)" in html
    assert "setProjection(&quot;orthographic&quot;)" in html
    assert "3dmol@2.5.5/build/3Dmol-min.js" in html
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
    assert "zoom(6.0)" in html
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
    assert "const panScale = 0.22" in html
    assert "pan(-event.deltaX * panScale, -event.deltaY * panScale)" in html
    assert '"not":{"atom":["N","CA","C","O","OXT"]}' in unescaped_html
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
    assert '"resi":[1],"resn":["DA","DC","DG","DT"]' not in unescaped_html
    assert '"resi":[1],"resn":["A","C","G","I","U"]' not in unescaped_html


def test_py3dmol_backend_uses_coordinate_derived_nucleic_ribbons_and_base_spokes() -> None:
    html = render_structure_view_html(
        StructureViewSpec(
            title="Default nucleic-acid context",
            models=(StructureViewModel("reference", _PROTEIN_AND_NUCLEIC_PDB, label="Reference"),),
        )
    )

    unescaped_html = html_lib.unescape(html).replace(" ", "")
    assert _DNA_RESIDUE_SELECTION in unescaped_html
    assert _RNA_RESIDUE_SELECTION in unescaped_html
    assert unescaped_html.count("addCustom(") == 2
    assert "addCurve(" not in unescaped_html
    assert unescaped_html.count("addCylinder(") == 6
    assert '"faceArr":' in unescaped_html
    assert '"vertexArr":' in unescaped_html
    assert '"color":"#B97700","opacity":1.0' in unescaped_html
    assert '"color":"#C84C5A","opacity":1.0' in unescaped_html
    assert '"radius":0.12,"fromCap":1,"toCap":1,"color":"#B97700"' in unescaped_html
    assert '"radius":0.12,"fromCap":1,"toCap":1,"color":"#C84C5A"' in unescaped_html
    assert '"representation":"backbone_ribbon_with_base_spokes"' in unescaped_html
    assert '"nucleotide_count":3' in unescaped_html
    assert '"base_spoke_count":3' in unescaped_html
    assert '"ribbon_mesh_count":1' in unescaped_html
    assert '"ribbon_vertex_count":12' in unescaped_html
    assert '"ribbon_triangle_count":20' in unescaped_html
    assert '"ribbon_width_angstrom":1.35' in unescaped_html
    assert '"ribbon_thickness_angstrom":0.28' in unescaped_html
    assert '"stick":{"color":"#B97700"' not in unescaped_html
    assert '"stick":{"color":"#C84C5A"' not in unescaped_html


def test_py3dmol_backend_rejects_nucleotide_geometry_without_c4_prime_anchors() -> None:
    incomplete_structure = _PROTEIN_AND_NUCLEIC_PDB.replace(" C4' ", " C5' ")

    with pytest.raises(ValueError, match="lacks C4-prime backbone anchor"):
        render_structure_view_html(
            StructureViewSpec(
                title="Incomplete nucleic geometry",
                models=(StructureViewModel("reference", incomplete_structure, label="Reference"),),
            )
        )


def test_py3dmol_backend_can_hide_nucleic_acid_classes() -> None:
    html = render_structure_view_html(
        StructureViewSpec(
            title="Hidden nucleic-acid context",
            models=(StructureViewModel("reference", _PROTEIN_AND_NUCLEIC_PDB, label="Reference"),),
            molecule_styles=(
                StructureViewMoleculeStyle("dna", "reference", label="DNA", color="#E69F00"),
                StructureViewMoleculeStyle("rna", "reference", label="RNA", color="#009E73"),
            ),
            hidden_molecule_classes=("dna",),
        )
    )

    unescaped_payload = html_lib.unescape(html)
    unescaped_html = unescaped_payload.replace(" ", "")
    assert "HETATM    6  P    DA B   1" not in unescaped_payload
    assert "HETATM    8  N1   DA B   1" not in unescaped_payload
    assert "HETATM   15  P     U C   2" in unescaped_payload
    assert "HETATM   17  N1    U C   2" in unescaped_payload
    assert '"resn":["DA","DC","DG","DT"]' not in unescaped_html
    assert '"resn":["A","C","G","I","U"]' in unescaped_html
    assert "#E69F00" not in html
    assert "DNA" not in html
    assert '"cartoon":{"color":"#E69F00"}' not in unescaped_html
    assert '"stick":{"color":"#E69F00","radius":0.18}' not in unescaped_html
    assert unescaped_html.count("addCustom(") == 1
    assert "addCurve(" not in unescaped_html
    assert unescaped_html.count("addCylinder(") == 3
    assert '"color":"#009E73","opacity":1.0' in unescaped_html
    assert '"radius":0.12,"fromCap":1,"toCap":1,"color":"#009E73"' in unescaped_html

    hidden_all_html = render_structure_view_html(
        StructureViewSpec(
            title="Hidden nucleic-acid context",
            models=(StructureViewModel("reference", _PROTEIN_AND_NUCLEIC_PDB, label="Reference"),),
            molecule_styles=(
                StructureViewMoleculeStyle("dna", "reference", label="DNA", color="#E69F00"),
                StructureViewMoleculeStyle("rna", "reference", label="RNA", color="#009E73"),
            ),
            hidden_molecule_classes=("dna", "rna"),
        )
    )

    hidden_all_payload = html_lib.unescape(hidden_all_html)
    hidden_all_unescaped = hidden_all_payload.replace(" ", "")
    assert "HETATM    6  P    DA B   1" not in hidden_all_payload
    assert "HETATM   15  P     U C   2" not in hidden_all_payload
    assert '"resn":["DA","DC","DG","DT"]' not in hidden_all_unescaped
    assert '"resn":["A","C","G","I","U"]' not in hidden_all_unescaped
    assert "addCustom(" not in hidden_all_unescaped
    assert "addCurve(" not in hidden_all_unescaped
    assert "addCylinder(" not in hidden_all_unescaped
    assert "#E69F00" not in hidden_all_html
    assert "#009E73" not in hidden_all_html


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
    assert '"atom":["O1P","O2P","OP1","OP2","P"]' not in unescaped_html
    assert '"resn":["ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",' in unescaped_html
    assert unescaped_html.count("addCustom(") == 2
    assert "addCurve(" not in unescaped_html
    assert unescaped_html.count("addCylinder(") == 6
    assert '"color":"#E69F00","opacity":1.0' in unescaped_html
    assert '"color":"#009E73","opacity":1.0' in unescaped_html


def test_py3dmol_backend_honors_explicit_nucleic_acid_stick_style() -> None:
    rendered = render_structure_view_html(
        StructureViewSpec(
            title="Nucleic-acid sticks",
            models=(StructureViewModel("reference", _PROTEIN_AND_NUCLEIC_PDB, label="Reference"),),
            molecule_styles=(
                StructureViewMoleculeStyle(
                    "dna",
                    "reference",
                    label="DNA",
                    color="#B97700",
                    style="stick",
                    radius=0.24,
                ),
                StructureViewMoleculeStyle(
                    "rna",
                    "reference",
                    label="RNA",
                    color="#C84C5A",
                    style="stick",
                    radius=0.24,
                ),
            ),
        )
    )

    unescaped = html_lib.unescape(rendered).replace(" ", "")
    assert '"stick":{"color":"#B97700","radius":0.24}' in unescaped
    assert '"stick":{"color":"#C84C5A","radius":0.24}' in unescaped
    assert "addCustom(" not in unescaped
    assert "addCurve(" not in unescaped
    assert '"style":"trace"' not in unescaped


def test_py3dmol_backend_accepts_explicit_nucleic_ribbon_with_base_spokes_style() -> None:
    rendered = render_structure_view_html(
        StructureViewSpec(
            title="Nucleic-acid traces and spokes",
            models=(StructureViewModel("reference", _PROTEIN_AND_NUCLEIC_PDB, label="Reference"),),
            molecule_styles=(
                StructureViewMoleculeStyle(
                    "dna",
                    "reference",
                    label="DNA",
                    color="#B97700",
                    style="backbone_ribbon_with_base_spokes",
                    width=1.5,
                    thickness=0.3,
                ),
                StructureViewMoleculeStyle(
                    "rna",
                    "reference",
                    label="RNA",
                    color="#C84C5A",
                    style="backbone_ribbon_with_base_spokes",
                    width=1.5,
                    thickness=0.3,
                ),
            ),
        )
    )

    unescaped = html_lib.unescape(rendered).replace(" ", "")
    assert unescaped.count("addCustom(") == 2
    assert "addCurve(" not in unescaped
    assert unescaped.count("addCylinder(") == 6
    assert '"color":"#B97700","opacity":1.0' in unescaped
    assert '"color":"#C84C5A","opacity":1.0' in unescaped
    assert '"radius":0.12,"fromCap":1,"toCap":1,"color":"#B97700"' in unescaped
    assert '"radius":0.12,"fromCap":1,"toCap":1,"color":"#C84C5A"' in unescaped
    assert '"ribbon_width_angstrom":1.5' in unescaped
    assert '"ribbon_thickness_angstrom":0.3' in unescaped


def test_py3dmol_backend_rejects_removed_nucleic_ribbon_style() -> None:
    with pytest.raises(ValueError, match="Unsupported molecule render style"):
        render_structure_view_html(
            StructureViewSpec(
                title="Removed nucleic style",
                models=(StructureViewModel("reference", _PROTEIN_AND_NUCLEIC_PDB, label="Reference"),),
                molecule_styles=(
                    StructureViewMoleculeStyle(
                        "dna",
                        "reference",
                        label="DNA",
                        color="#B97700",
                        style="backbone_ribbon_with_base_sticks",  # type: ignore[arg-type]
                    ),
                ),
            )
        )


@pytest.mark.parametrize("style", ["cartoon", "surface"])
def test_py3dmol_backend_rejects_nucleic_styles_that_create_slabs_or_occlusion(style: str) -> None:
    with pytest.raises(ValueError, match="DNA and RNA styles"):
        render_structure_view_html(
            StructureViewSpec(
                title="Invalid nucleic style",
                models=(StructureViewModel("reference", _PROTEIN_AND_NUCLEIC_PDB, label="Reference"),),
                molecule_styles=(
                    StructureViewMoleculeStyle(
                        "dna",
                        "reference",
                        label="DNA",
                        color="#B97700",
                        style=style,
                    ),
                ),
            )
        )


def test_py3dmol_backend_colors_selected_nucleic_acid_spokes_without_atom_sticks() -> None:
    dna_html = render_structure_view_html(
        StructureViewSpec(
            title="DNA-scoped selection",
            models=(StructureViewModel("reference", _PROTEIN_AND_NUCLEIC_PDB, label="Reference"),),
            selection_styles=(
                StructureViewSelectionStyle(
                    selection_id="dna_site",
                    model_id="reference",
                    label="DNA residue 1",
                    residue_numbers=(1,),
                    residue_scope="dna",
                    color="#D55E00",
                ),
            ),
        )
    )

    unescaped_dna = html_lib.unescape(dna_html).replace(" ", "")
    assert "DNA residue 1" in dna_html
    assert '"resi":[1],"resn":["DA","DC","DG","DT"]' in unescaped_dna
    assert '"atom":["O1P","O2P","OP1","OP2","P"]' not in unescaped_dna
    assert '"radius":0.15,"fromCap":1,"toCap":1,"color":"#D55E00"' in unescaped_dna
    assert '"stick":{"color":"#D55E00"' not in unescaped_dna

    rna_html = render_structure_view_html(
        StructureViewSpec(
            title="RNA-scoped selection",
            models=(StructureViewModel("reference", _PROTEIN_AND_NUCLEIC_PDB, label="Reference"),),
            selection_styles=(
                StructureViewSelectionStyle(
                    selection_id="rna_site",
                    model_id="reference",
                    label="RNA residue 2",
                    residue_numbers=(2,),
                    residue_scope="rna",
                    color="#D55E00",
                ),
            ),
        )
    )

    unescaped_rna = html_lib.unescape(rna_html).replace(" ", "")
    assert "RNA residue 2" in rna_html
    assert '"resi":[2],"resn":["A","C","G","I","U"]' in unescaped_rna
    assert '"atom":["O1P","O2P","OP1","OP2","P"]' not in unescaped_rna
    assert '"radius":0.15,"fromCap":1,"toCap":1,"color":"#D55E00"' in unescaped_rna
    assert '"stick":{"color":"#D55E00"' not in unescaped_rna


def test_py3dmol_backend_colors_visible_protein_sidechains_with_molecule_class() -> None:
    html = render_structure_view_html(
        StructureViewSpec(
            title="Protein-class coloring",
            models=(
                StructureViewModel(
                    "reference",
                    _SIDECHAIN_PDB,
                    label="Reference",
                    color="#8c959f",
                    show_sidechains=True,
                    sidechain_color="#8c959f",
                ),
            ),
            molecule_styles=(StructureViewMoleculeStyle("protein", "reference", label="Protein", color="#0072B2"),),
        )
    )

    unescaped_html = html_lib.unescape(html).replace(" ", "")
    assert '"not":{"atom":["N","CA","C","O","OXT"]}' in unescaped_html
    assert '"cartoon":{"color":"#0072B2"}' in unescaped_html
    assert '"stick":{"color":"#0072B2","radius":0.16}' in unescaped_html
    assert '"stick":{"color":"#8c959f","radius":0.16}' not in unescaped_html


def test_py3dmol_backend_applies_selection_after_visible_sidechain_styles() -> None:
    html = render_structure_view_html(
        StructureViewSpec(
            title="Selected side-chain coloring",
            models=(
                StructureViewModel(
                    "reference",
                    _SIDECHAIN_PDB,
                    label="Reference",
                    color="#0072B2",
                    show_sidechains=True,
                ),
            ),
            selection_styles=(
                StructureViewSelectionStyle(
                    selection_id="active_site",
                    model_id="reference",
                    label="Active-site residues",
                    residue_numbers=(1,),
                    color="#D55E00",
                    show_sidechains=True,
                ),
            ),
            hidden_molecule_classes=("dna", "rna"),
        )
    )

    unescaped_html = html_lib.unescape(html).replace(" ", "")
    base_sticks = '"stick":{"color":"#0072B2","radius":0.16}'
    highlighted_sticks = '"stick":{"color":"#D55E00","radius":0.22}'
    assert base_sticks in unescaped_html
    assert highlighted_sticks in unescaped_html
    assert unescaped_html.index(base_sticks) < unescaped_html.index(highlighted_sticks)


def test_py3dmol_backend_hides_selected_protein_sidechains_when_sidechains_are_disabled() -> None:
    html = render_structure_view_html(
        StructureViewSpec(
            title="Selected side-chain-only emphasis",
            models=(StructureViewModel("reference", _SIDECHAIN_PDB, label="Reference", color="#F7F3EA"),),
            selection_styles=(
                StructureViewSelectionStyle(
                    selection_id="active_site",
                    model_id="reference",
                    label="Active-site residues",
                    residue_numbers=(1,),
                    color="#C00000",
                ),
            ),
            hidden_molecule_classes=("dna", "rna"),
        )
    )

    unescaped_html = html_lib.unescape(html).replace(" ", "")
    assert '"stick":{"color":"#F7F3EA","radius":0.16}' not in unescaped_html
    assert '"stick":{"color":"#C00000","radius":0.22}' not in unescaped_html


def test_py3dmol_backend_can_emit_explicit_molecule_surface_style() -> None:
    html = render_structure_view_html(
        StructureViewSpec(
            title="Protein surface style",
            models=(StructureViewModel("reference", _SIDECHAIN_PDB, label="Reference"),),
            molecule_styles=(
                StructureViewMoleculeStyle(
                    "protein",
                    "reference",
                    label="Protein surface",
                    color="#D55E00",
                    opacity=0.42,
                    style="surface",
                ),
            ),
        )
    )

    unescaped_html = html_lib.unescape(html).replace(" ", "")
    assert 'addSurface("VDW",{"color":"#D55E00","opacity":0.42}' in unescaped_html
    assert '"resn":["ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",' in unescaped_html


def test_py3dmol_backend_keeps_surface_highlight_at_the_declared_surface_alpha() -> None:
    html = render_structure_view_html(
        StructureViewSpec(
            title="Protein surface selection",
            models=(StructureViewModel("reference", _SIDECHAIN_PDB, label="Reference"),),
            molecule_styles=(
                StructureViewMoleculeStyle(
                    "protein",
                    "reference",
                    label="Protein surface",
                    color="#F7F3EA",
                    opacity=0.78,
                    style="surface",
                ),
            ),
            selection_styles=(
                StructureViewSelectionStyle(
                    selection_id="contact_site",
                    model_id="reference",
                    label="Contact site",
                    residue_numbers=(1,),
                    color="#C00000",
                    show_sidechains=True,
                ),
            ),
        )
    )

    unescaped_html = html_lib.unescape(html).replace(" ", "")
    assert 'addSurface("VDW",{"color":"#F7F3EA","opacity":0.78}' in unescaped_html
    assert 'addSurface("VDW",{"color":"#C00000","opacity":0.78}' in unescaped_html
    assert '"stick":{"color":"#C00000","radius":0.22}' in unescaped_html


def test_py3dmol_backend_serializes_mmcif_for_3dmol_without_prime_token_ambiguity() -> None:
    html = render_structure_view_html(
        StructureViewSpec(
            title="CIF reference review",
            models=(
                StructureViewModel(
                    "reference",
                    _MIXED_POLYMER_MMCIF_WITH_UNQUOTED_PRIME_ATOMS,
                    structure_format="mmcif",
                    label="Reference",
                    color="#d6d6d6",
                    show_sidechains=True,
                ),
            ),
        )
    )

    unescaped_html = html_lib.unescape(html)
    assert 'addModel("data_dnadesign_browser\\nloop_\\n_atom_site.group_pdb' in unescaped_html
    assert '\\"O5\'\\"' in unescaped_html
    assert '\\"C4\'\\"' in unescaped_html
    assert "data_mixed_polymer" not in unescaped_html
    assert '","cif");' in unescaped_html
    assert '","mmcif");' not in unescaped_html


@pytest.mark.parametrize(
    ("structure_text", "hidden_molecule_classes"),
    (
        (_DNA_ONLY_MMCIF, ("dna",)),
        (_PROTEIN_ONLY_MMCIF, ("protein",)),
        (_SIDECHAIN_MMCIF, ("protein", "dna", "rna")),
        (_DNA_ONLY_MMCIF_WITH_ENTITY_CATEGORY, ("dna",)),
        (_DNA_ONLY_MULTIMODEL_MMCIF, ("dna",)),
    ),
)
def test_py3dmol_backend_renders_empty_mmcif_model_when_filters_hide_every_atom(
    structure_text: str,
    hidden_molecule_classes: tuple[str, ...],
) -> None:
    rendered = render_structure_view_html(
        StructureViewSpec(
            title="Hidden mmCIF model",
            models=(
                StructureViewModel(
                    "reference",
                    structure_text,
                    structure_format="mmcif",
                    label="Reference",
                ),
            ),
            hidden_molecule_classes=hidden_molecule_classes,
        )
    )

    unescaped_html = html_lib.unescape(rendered)
    assert unescaped_html.count("addModel(") == 1
    assert 'addModel("data_dnadesign_browser\\nloop_\\n_atom_site.group_pdb' in unescaped_html
    assert "ATOM 1" not in unescaped_html
    assert "HETATM 6" not in unescaped_html
    assert "_entity.type" not in unescaped_html
    assert '","cif");' in unescaped_html


def test_py3dmol_backend_preserves_unclassified_ligand_when_all_known_classes_are_hidden() -> None:
    rendered = render_structure_view_html(
        StructureViewSpec(
            title="Visible ligand",
            models=(
                StructureViewModel(
                    "reference",
                    _LIGAND_ONLY_MMCIF,
                    structure_format="mmcif",
                    label="Reference",
                ),
            ),
            hidden_molecule_classes=("protein", "dna", "rna"),
        )
    )

    assert "HEM" in html_lib.unescape(rendered)


def test_py3dmol_backend_preserves_model_indices_when_one_mmcif_model_is_filtered_empty() -> None:
    rendered = render_structure_view_html(
        StructureViewSpec(
            title="Mixed empty and visible models",
            models=(
                StructureViewModel(
                    "hidden_dna",
                    _DNA_ONLY_MMCIF,
                    structure_format="mmcif",
                    label="Hidden DNA",
                ),
                StructureViewModel(
                    "visible_protein",
                    _PROTEIN_ONLY_MMCIF,
                    structure_format="mmcif",
                    label="Visible protein",
                ),
            ),
            hidden_molecule_classes=("dna",),
        )
    )

    unescaped_html = html_lib.unescape(rendered)
    assert unescaped_html.count("addModel(") == 2
    assert "HETATM 6" not in unescaped_html
    assert "ATOM 1" in unescaped_html
    assert '"model":1' in unescaped_html.replace(" ", "")


def test_py3dmol_backend_empty_filter_matches_pdb_behavior() -> None:
    dna_only_pdb = """\
HETATM    1  P    DA A   1       0.000   0.000   0.000  1.00 80.00           P
END
"""

    for structure_text, structure_format in (
        (dna_only_pdb, "pdb"),
        (_DNA_ONLY_MMCIF, "mmcif"),
    ):
        rendered = render_structure_view_html(
            StructureViewSpec(
                title=f"Hidden {structure_format} model",
                models=(
                    StructureViewModel(
                        "reference",
                        structure_text,
                        structure_format=structure_format,
                        label="Reference",
                    ),
                ),
                hidden_molecule_classes=("dna",),
            )
        )
        assert html_lib.unescape(rendered).count("addModel(") == 1


def test_mmcif_serializer_still_rejects_source_with_native_zero_row_coordinate_loop() -> None:
    with pytest.raises(ValueError, match="requires at least one atom-site record"):
        serialize_mmcif_atom_sites_for_3dmol(_EMPTY_ATOM_SITE_MMCIF)


def test_py3dmol_backend_does_not_let_empty_filter_hide_malformed_mmcif_columns() -> None:
    malformed = _DNA_ONLY_MMCIF.replace("_atom_site.Cartn_z\n", "")

    with pytest.raises(ValueError, match="_atom_site.cartn_z"):
        render_structure_view_html(
            StructureViewSpec(
                title="Malformed hidden mmCIF model",
                models=(
                    StructureViewModel(
                        "reference",
                        malformed,
                        structure_format="mmcif",
                        label="Reference",
                    ),
                ),
                hidden_molecule_classes=("dna",),
            )
        )


def test_py3dmol_backend_uses_single_quotes_for_double_quote_atom_identifiers() -> None:
    structure = _SIDECHAIN_MMCIF.replace("ATOM 3 C CB .", "ATOM 3 C 'C\"1' .")

    serialized = serialize_mmcif_atom_sites_for_3dmol(structure)

    assert "'C\"1'" in serialized


def test_py3dmol_backend_preserves_unquoted_atom_site_null_markers() -> None:
    serialized = serialize_mmcif_atom_sites_for_3dmol(_SIDECHAIN_MMCIF)

    assert 'ATOM 1 N "N" . SER A 3 0.000 0.000 0.000 A 3 ? 1.00 80.00 1' in serialized
    assert '"."' not in serialized
    assert '"?"' not in serialized


def test_py3dmol_backend_preserves_quoted_literal_atom_site_null_tokens() -> None:
    structure = _SIDECHAIN_MMCIF.replace("ATOM 1 N N . SER", "ATOM 1 N N '.' SER", 1).replace(
        "A 3 ? 1.00 80.00 1",
        "A 3 '?' 1.00 80.00 1",
        1,
    )

    serialized = serialize_mmcif_atom_sites_for_3dmol(structure)

    assert 'ATOM 1 N "N" "." SER A 3 0.000 0.000 0.000 A 3 "?" 1.00 80.00 1' in serialized


@pytest.mark.parametrize("value", [".", "?"])
def test_py3dmol_backend_preserves_semicolon_text_null_like_literals(value: str) -> None:
    structure = _SIDECHAIN_MMCIF.replace(
        "ATOM 1 N N . SER",
        f"ATOM 1 N N\n;{value}\n;\nSER",
        1,
    )

    serialized = serialize_mmcif_atom_sites_for_3dmol(structure)

    assert f'ATOM 1 N "N" "{value}" SER A 3 0.000 0.000 0.000 A 3 ? 1.00 80.00 1' in serialized


def test_py3dmol_backend_uses_concrete_auth_atom_name_for_unquoted_label_null() -> None:
    serialized = serialize_mmcif_atom_sites_for_3dmol(_ATOM_NAME_FALLBACK_MMCIF)

    assert 'ATOM 1 C "CA" . SER A 3 1.000 1.000 0.000 A 3 ? 1.00 80.00 1' in serialized
    assert 'ATOM 2 C "CB" . SER A 3 2.000 1.000 0.000 A 3 ? 1.00 80.00 1' in serialized
    assert '"AUTH"' not in serialized


def test_py3dmol_backend_preserves_source_quoted_literal_label_atom_name() -> None:
    structure = _ATOM_NAME_FALLBACK_MMCIF.replace("ATOM 1 C . CA .", "ATOM 1 C '.' CA .")

    serialized = serialize_mmcif_atom_sites_for_3dmol(structure)

    assert 'ATOM 1 C "." . SER A 3 1.000 1.000 0.000 A 3 ? 1.00 80.00 1' in serialized


def test_py3dmol_backend_rejects_null_label_and_auth_atom_names() -> None:
    structure = _ATOM_NAME_FALLBACK_MMCIF.replace("ATOM 1 C . CA .", "ATOM 1 C . ? .")

    with pytest.raises(ValueError, match="concrete label_atom_id or auth_atom_id"):
        serialize_mmcif_atom_sites_for_3dmol(structure)


def test_mmcif_token_serializer_rejects_null_marker_with_unknown_quote_provenance() -> None:
    with pytest.raises(ValueError, match="quote provenance is unknown"):
        _browser_cif_token(".", field="_atom_site.label_alt_id", source_quoted=None)


def test_py3dmol_backend_rejects_atom_identifiers_with_both_quote_delimiters() -> None:
    with pytest.raises(ValueError, match="atom name cannot be serialized safely"):
        _quote_3dmol_atom_name("C'\"1")


def test_py3dmol_backend_quotes_whitespace_bearing_mmcif_identifiers() -> None:
    structure = _SIDECHAIN_MMCIF.replace(" SER A 1 3 ", " SER 'chain A' 1 3 ").replace(
        " 0.000 A 3 ? ",
        " 0.000 'chain A' 3 ? ",
    )

    serialized = serialize_mmcif_atom_sites_for_3dmol(structure)

    assert '"chain A"' in serialized


@pytest.mark.parametrize(
    "identifier",
    (
        ".",
        "?",
        "#chain",
        "_chain",
        "$chain",
        "[chain",
        "]chain",
        "data_chain",
        "DaTa_chain",
        "save_chain",
        "SaVe_chain",
        "STOP_",
        "global_",
    ),
)
def test_py3dmol_backend_quotes_reserved_mmcif_identifiers(identifier: str) -> None:
    structure = _SIDECHAIN_MMCIF.replace(" SER A 1 3 ", f" SER '{identifier}' 1 3 ").replace(
        " 0.000 A 3 ? ",
        f" 0.000 '{identifier}' 3 ? ",
    )

    serialized = serialize_mmcif_atom_sites_for_3dmol(structure)

    assert f'"{identifier}"' in serialized


@pytest.mark.parametrize("value", ("loop_", "LOOP_", "stop_", "global_"))
def test_mmcif_token_serializer_quotes_reserved_control_words(value: str) -> None:
    assert _browser_cif_token(value, field="_atom_site.auth_asym_id") == f'"{value}"'


@pytest.mark.parametrize("value", (".", "?"))
def test_mmcif_token_serializer_quotes_source_quoted_null_literals(value: str) -> None:
    assert _browser_cif_token(value, field="_atom_site.auth_asym_id", source_quoted=True) == f'"{value}"'


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


def test_structure_molecule_class_summary_detects_protein_dna_and_rna() -> None:
    assert molecule_classes_in_structure_text(
        _PROTEIN_AND_NUCLEIC_PDB,
        structure_format="pdb",
    ) == frozenset({"protein", "dna", "rna"})


def test_structure_text_filter_keeps_only_requested_molecule_roles() -> None:
    nucleic_text = filter_structure_text_by_molecule_classes(
        _PROTEIN_AND_NUCLEIC_PDB,
        structure_format="pdb",
        visible_molecule_classes=("dna", "rna"),
    )

    assert " ALA A " not in nucleic_text
    assert " DA B " in nucleic_text
    assert " U C " in nucleic_text
    assert molecule_classes_in_structure_text(
        nucleic_text,
        structure_format="pdb",
    ) == frozenset({"dna", "rna"})


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


def test_structure_view_contract_rejects_duplicate_model_ids() -> None:
    with pytest.raises(ValueError, match="model_id values must be unique"):
        StructureViewSpec(
            title="Duplicate model identifiers",
            models=(
                StructureViewModel("shared", _MINIMAL_PDB, label="Reference"),
                StructureViewModel("shared", _SIDECHAIN_PDB, label="Query"),
            ),
        ).validate()


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
