"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/notebook_assertions.py

Notebook contract assertions for Eco1 review-deliverable tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from dnadesign.studies.units.eco1_rt_repack.tests._helpers import repo_root
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.runtime_fixtures import (
    resolve_manifest_path,
)


def assert_manifest_visual_contract(
    *,
    manifest_path: Path,
    manifest: dict[str, Any],
    deliverables: dict[str, dict[str, Any]],
    expected_rendered: set[str],
) -> None:
    """Assert rendered deliverables have portable paths and accessible visual metadata."""

    for deliverable in manifest["deliverables"]:
        assert not Path(deliverable["path"]).is_absolute()
        assert deliverable["title"].strip()
        assert not deliverable["title"].rstrip().endswith(".")
        assert deliverable["alt_text"].strip()
        assert deliverable["description"].strip()
        assert deliverable["interpretation_limit"].strip()
        assert deliverable["source_tables"]
        assert deliverable["input_hashes"]

    for deliverable_id in expected_rendered:
        path = resolve_manifest_path(manifest_path, deliverables[deliverable_id]["path"])
        assert path.exists(), deliverable_id
        if path.suffix == ".svg":
            svg_text = path.read_text(encoding="utf-8")
            svg_root = ET.parse(path).getroot()
            assert "<title" in svg_text
            assert "<desc" in svg_text
            assert svg_root.findall(".//{http://www.w3.org/2000/svg}text")
            assert "Ec86 clade 9 MSA plurality and mask context" not in svg_text


def assert_review_notebook_contract(notebook_text: str) -> None:
    """Assert the generated marimo notebook stays manifest-backed and plain."""

    runtime_path = repo_root() / (
        "src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/notebook_runtime.py"
    )
    runtime_dir = runtime_path.parent
    runtime_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            runtime_path,
            runtime_dir / "notebook_sae_features.py",
            runtime_dir / "notebook_structure_browser.py",
        )
    )
    combined_text = notebook_text + "\n" + runtime_text
    assert 'marimo.App(width="medium")' in notebook_text
    assert "notebook_runtime import" in notebook_text
    assert "review_deliverable_manifest.yaml" in combined_text
    assert "manifest_root = manifest_path.parent" in combined_text
    assert "def resolve_manifest_path(" in runtime_text
    assert "_resolve_manifest_path(" not in notebook_text
    assert "deliverable_section_ui = mo.ui.dropdown" in notebook_text
    assert "deliverable_id_ui = mo.ui.dropdown" in notebook_text
    assert "review_lane_ui = mo.ui.dropdown" in notebook_text
    assert 'label="Deliverable lane"' in notebook_text
    assert "selected_deliverable(" in notebook_text
    assert "sections: list[str] = []" in runtime_text
    assert 'sorted({str(row["section"])' not in notebook_text
    assert "Repacking Eco1 reverse transcriptase for structured-template assays" in combined_text
    assert "Tao-style fixed-backbone" in combined_text
    assert "repack the remaining design canvas" in combined_text
    assert "Mestre-derived clade 9" in combined_text
    assert "active mask uses" in combined_text
    assert "WT ESMC" in combined_text
    assert "review-only model-constraint audit" in combined_text
    assert "Biohub ESMC SAE features annotate WT" in combined_text
    assert "alignments and ESMC" not in combined_text
    assert "ProteinMPNN proposes variants" in combined_text
    assert "unprotected" in combined_text
    assert "review surface follows" not in combined_text
    assert "does not rerun ProteinMPNN" not in combined_text
    assert "This study asks" not in combined_text
    assert "review surface" not in combined_text
    assert "Eco1/Ec86 is a retron reverse transcriptase with a cryoEM-supported scaffold.\\n" not in combined_text
    assert "ProteinMPNN samples the mutable canvas. ColabFold checks\\n" not in combined_text
    assert "white-space:normal" in combined_text
    assert "max-width:76ch" not in combined_text
    assert "Deliverables:" not in combined_text
    assert "Status:" not in combined_text
    assert "status_summary_text" not in combined_text
    assert "Analysis section" in notebook_text
    assert 'label="Figure / browser"' in notebook_text
    assert "mo.hstack(" in notebook_text
    assert "[review_lane_ui, deliverable_section_ui, deliverable_id_ui]" in notebook_text.replace("\n", "")
    assert "section_deliverables" in combined_text
    assert "mo.accordion(visual_panels, multiple=False, lazy=True)" not in combined_text
    assert "format_section_label(" in combined_text
    assert "format_deliverable_label(" in combined_text
    assert 'str(row.get("title") or "")' in runtime_text
    assert "Constraint evidence for the design mask" in combined_text
    assert "ProteinMPNN designs and fold triage" in combined_text
    assert "ESMC feature review" in combined_text
    assert "Feasibility and handoff" in combined_text
    assert "Planned downstream gate" in combined_text
    assert "biohub_esmc_protein_top_sae_features" in combined_text
    assert "Protein" in notebook_text
    assert "WT Ec86 control" in combined_text
    assert "SAE feature" in notebook_text
    assert "is_interactive_structure_deliverable(" in combined_text
    assert "render_deliverable_details(" in combined_text
    assert 'if selected_section == "fold_review":' not in notebook_text
    assert "No source-backed description is available for this exact SAE dictionary" in combined_text
    assert "Reference sequence, alignment, and mask" not in combined_text
    assert "Reference scaffold and mask evidence" not in combined_text
    assert "ProteinMPNN sequence proposals" not in combined_text
    assert "ColabFold structure triage" not in combined_text
    assert 'structure_label = "Structure view"' in notebook_text
    assert "label=structure_label" in notebook_text
    assert "structure_group_ui = mo.ui.dropdown" in notebook_text
    assert 'label="Structure group"' in notebook_text
    assert "structure_background_ui = mo.ui.checkbox" in notebook_text
    assert 'label="Reference background"' in notebook_text
    assert "structure_mutation_ui = mo.ui.checkbox" in notebook_text
    assert 'label="Mutation differences"' in notebook_text
    assert "selected_deliverable_id=selected_visual_id" in notebook_text
    assert "selected_group=selected_structure_group" in notebook_text
    assert "structure_group_lookup" in combined_text
    assert "_NOTEBOOK_HIDDEN_DELIVERABLE_IDS" in combined_text
    assert "LLR = log P(alternate) - log P(WT)" in combined_text
    assert "Method and row counts" in combined_text
    assert "visual_deliverables" in combined_text
    assert "Section deliverables" not in combined_text
    assert "Additional visuals in this section" not in combined_text
    assert 'mo.md("## All visuals in this section")' not in combined_text
    assert "selected_title =" not in combined_text
    assert 'mo.md(f"## {selected_title}")' not in combined_text
    assert "mask_structure_context_script" not in notebook_text
    assert "mask_structure_context_orientation_template" not in notebook_text
    assert "structure_overlay_skipped" not in notebook_text
    assert "render_deliverable_artifact(" in runtime_text
    assert "render_interpretation_note(" in combined_text
    assert "<strong>Interpretation limit:</strong>" in combined_text
    assert "overflow-x:auto" not in combined_text
    assert "wide_visual" in combined_text
    assert "render_mode" in combined_text
    assert "is_wide_visual" in combined_text
    assert "width:100%" in combined_text
    assert "max-width:100%" in combined_text
    assert "Interpretation limit" in combined_text
    assert "\n    deliverable_section_ui\n" not in notebook_text
    for cell in notebook_text.split("@app.cell"):
        if "deliverable_section_ui = mo.ui.dropdown(" in cell:
            assert ".value" not in cell


def assert_chimerax_context_scripts(
    *,
    manifest_path: Path,
    deliverables: dict[str, dict[str, Any]],
    forbidden_path_text: str,
) -> None:
    """Assert ChimeraX review scripts are portable and preserve the mask-context contract."""

    chimerax_text = resolve_manifest_path(
        manifest_path,
        deliverables["mask_structure_context_script"]["path"],
    ).read_text(encoding="utf-8")
    assert "eco1_rt_clade9_plurality25_direct_contact5a_v1" in chimerax_text
    assert "set bgColor white" in chimerax_text
    assert "camera ortho" in chimerax_text
    assert '2dlabels text "Ec86 reference"' in chimerax_text
    assert "view orient" in chimerax_text
    assert "# orientation_preset_id: ec86_reference_thumb_down_v1" in chimerax_text
    assert "design canvas" in chimerax_text
    assert "color" in chimerax_text
    assert forbidden_path_text not in chimerax_text

    orientation_text = resolve_manifest_path(
        manifest_path,
        deliverables["mask_structure_context_orientation_template"]["path"],
    ).read_text(encoding="utf-8")
    assert "Manual orientation handoff" in orientation_text
    assert "save mask_structure_context_orientation.cxs" in orientation_text
    assert "exit" not in orientation_text
    assert forbidden_path_text not in orientation_text
