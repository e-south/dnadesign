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
        path = _resolve_manifest_path(manifest_path, deliverables[deliverable_id]["path"])
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
    assert "selected_deliverable(" in notebook_text
    assert "sections: list[str] = []" in runtime_text
    assert 'sorted({str(row["section"])' not in notebook_text
    assert "Repacking Eco1 reverse transcriptase while preserving the RT scaffold" in combined_text
    assert "Tao-style fixed-backbone" in combined_text
    assert "redesign can repack" in combined_text
    assert "Mestre-derived clade 9" in combined_text
    assert "alignments and ESMC" in combined_text
    assert "ProteinMPNN proposes sequences" in combined_text
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
    assert 'label="Visual"' in notebook_text
    assert "mo.hstack([deliverable_section_ui, deliverable_id_ui]" in notebook_text
    assert "section_deliverables" in combined_text
    assert "mo.accordion(visual_panels, multiple=False, lazy=True)" not in combined_text
    assert "format_section_label(" in combined_text
    assert "format_deliverable_label(" in combined_text
    assert 'str(row.get("title") or "")' in runtime_text
    assert "WT ESMC substitution constraint" in combined_text
    assert "Biohub ESMC SAE interpretation" in combined_text
    assert "biohub_esmc_protein_top_sae_features" in combined_text
    assert "Protein" in notebook_text
    assert "WT Ec86 control" in combined_text
    assert "SAE feature" in notebook_text
    assert "No source-backed description is available for this exact SAE dictionary" in combined_text
    assert "Reference sequence, alignment, and mask" in combined_text
    assert "Reference scaffold and mask evidence" not in combined_text
    assert "ProteinMPNN sequence proposals" in combined_text
    assert "ColabFold structure triage" in combined_text
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
    assert "overflow-x:auto" not in combined_text
    assert "is_wide = aspect_ratio >= 4.0" not in combined_text
    assert "width:100%" in combined_text
    assert "max-width:100%" in combined_text
    assert "Interpretation limit" in combined_text
    assert "\n    deliverable_section_ui\n" not in notebook_text
    for cell in notebook_text.split("@app.cell"):
        if "deliverable_section_ui = mo.ui.dropdown(" in cell:
            assert ".value" not in cell


def _visual_option_source(notebook_text: str) -> str:
    marker = "visual_deliverables = ["
    start = notebook_text.find(marker)
    assert start != -1
    end = notebook_text.find("return", start)
    assert end != -1
    return notebook_text[start:end]


def _resolve_manifest_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest_path.parent / path
