"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/test_materialization.py

Eco1 review-deliverable materialization tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import yaml
from pytest import MonkeyPatch

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables import (
    mask_tracks,
    materialize_review_deliverables,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.fixtures import (
    write_deliverable_inputs,
)


def test_review_deliverables_materialize_manifest_figures_and_notebook(tmp_path: Path) -> None:
    write_deliverable_inputs(tmp_path)

    result = materialize_review_deliverables(repo_root=Path.cwd(), output_root=tmp_path, render_chimerax_png=False)

    manifest = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_id"] == "eco1_rt.review_deliverables"
    assert manifest["status"] == "materialized"
    assert manifest["deliverable_count"] == len(manifest["deliverables"])
    assert manifest["visual_policy"]["requires_alt_text"] is True
    assert manifest["notebook"]["path"] == "notebooks/eco1_review_deliverables.py"
    assert not Path(manifest["notebook"]["path"]).is_absolute()

    deliverables = {entry["deliverable_id"]: entry for entry in manifest["deliverables"]}
    expected_rendered = {
        "msa_plurality_mask_panel",
        "linear_mask_tracks",
        "proteinmpnn_score_mutation_burden",
        "proteinmpnn_mutation_density",
        "mask_structure_context_script",
        "mask_structure_context_orientation_template",
    }
    assert expected_rendered.issubset(deliverables)
    assert deliverables["mask_structure_context_png"]["status"] == "skipped_optional_render_disabled"
    assert deliverables["foldcheck_review_fold_metric_scatter"]["status"] == "linked_existing"

    for deliverable in manifest["deliverables"]:
        assert not Path(deliverable["path"]).is_absolute()
        assert deliverable["alt_text"].strip()
        assert deliverable["description"].strip()
        assert deliverable["interpretation_limit"].strip()
        assert deliverable["source_tables"]
        assert deliverable["input_hashes"]

    for deliverable_id in expected_rendered:
        path = _resolve_manifest_path(result.manifest_path, deliverables[deliverable_id]["path"])
        assert path.exists(), deliverable_id
        if path.suffix == ".svg":
            svg_text = path.read_text(encoding="utf-8")
            svg_root = ET.parse(path).getroot()
            assert "<title" in svg_text
            assert "<desc" in svg_text
            assert svg_root.findall(".//{http://www.w3.org/2000/svg}text")
            assert "Ec86 clade 9 MSA plurality and mask context" not in svg_text

    msa_text = _resolve_manifest_path(
        result.manifest_path,
        deliverables["msa_plurality_mask_panel"]["path"],
    ).read_text(encoding="utf-8")
    assert "Clade 9 alignment shows which Ec86 positions were protected." in msa_text
    assert "ec86_clade9_conservation_v1__" not in msa_text
    assert "25% WT plurality" in msa_text

    diversity_text = _resolve_manifest_path(
        result.manifest_path,
        deliverables["proteinmpnn_score_mutation_burden"]["path"],
    ).read_text(encoding="utf-8")
    assert "ProteinMPNN sampled two temperature settings." in diversity_text
    assert "Sequence identity to Ec86 WT (%)" in diversity_text
    assert "Accepted designs retain a minority of WT residues." not in diversity_text
    assert "Sampling temperature" not in diversity_text

    linked_fold_plot = _resolve_manifest_path(
        result.manifest_path,
        deliverables["foldcheck_review_fold_metric_scatter"]["path"],
    )
    assert linked_fold_plot.exists()
    assert linked_fold_plot.parent.name == "plots"

    chimerax_text = _resolve_manifest_path(
        result.manifest_path,
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
    assert str(tmp_path) not in chimerax_text

    orientation_text = _resolve_manifest_path(
        result.manifest_path,
        deliverables["mask_structure_context_orientation_template"]["path"],
    ).read_text(encoding="utf-8")
    assert "Manual orientation handoff" in orientation_text
    assert "save mask_structure_context_orientation.cxs" in orientation_text
    assert "exit" not in orientation_text
    assert str(tmp_path) not in orientation_text

    notebook_text = result.notebook_path.read_text(encoding="utf-8")
    assert 'marimo.App(width="medium")' in notebook_text
    assert "review_deliverable_manifest.yaml" in notebook_text
    assert "manifest_root = manifest_path.parent" in notebook_text
    assert "def resolve_manifest_path(" in notebook_text
    assert "_resolve_manifest_path(" not in notebook_text
    assert "deliverable_section_ui = mo.ui.dropdown" in notebook_text
    assert "deliverable_id_ui = mo.ui.dropdown" in notebook_text
    assert "sections = []" in notebook_text
    assert 'sorted({str(row["section"])' not in notebook_text
    assert "Repacking the Eco1 reverse transcriptase" in notebook_text
    assert "ProteinMPNN proposes sequence variants" in notebook_text
    assert "Deliverables:" not in notebook_text
    assert "Status:" not in notebook_text
    assert "status_summary_text" not in notebook_text
    assert "Review section" in notebook_text
    assert "Visual" in notebook_text
    assert "mo.hstack" in notebook_text
    assert "format_section_label(" in notebook_text
    assert "format_deliverable_label(" in notebook_text
    assert "visual_deliverables" in notebook_text
    assert "Section deliverables" not in notebook_text
    assert "Additional visuals in this section" not in notebook_text
    assert 'mo.md("## All visuals in this section")' not in notebook_text
    assert "mask_structure_context_script" not in _visual_option_source(notebook_text)
    assert "mask_structure_context_orientation_template" not in _visual_option_source(notebook_text)
    assert "render_deliverable_artifact(" in notebook_text
    assert "image_aspect_ratio(" in notebook_text
    assert "overflow-x:auto" in notebook_text
    assert "width:min(" in notebook_text
    assert "max-width:100%" in notebook_text
    assert "max-width:none" not in notebook_text
    assert "Interpretation limit" in notebook_text
    assert "\n    deliverable_section_ui\n" not in notebook_text
    assert "\n    deliverable_id_ui\n" not in notebook_text
    for cell in notebook_text.split("@app.cell"):
        if "deliverable_section_ui = mo.ui.dropdown(" in cell:
            assert ".value" not in cell
        if "deliverable_id_ui = mo.ui.dropdown(" in cell:
            assert ".value" not in cell


def test_chimerax_render_uses_gui_backed_script_mode(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    script_path = tmp_path / "mask_structure_context.cxc"
    script_path.write_text("exit\n", encoding="utf-8")
    recorded_args: list[str] = []

    def fake_run(args: list[str], **_kwargs: object) -> SimpleNamespace:
        recorded_args.extend(args)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(mask_tracks.subprocess, "run", fake_run)

    assert mask_tracks._run_chimerax(
        executable="/Applications/ChimeraX.app/Contents/MacOS/ChimeraX", script_path=script_path
    )
    assert recorded_args == [
        "/Applications/ChimeraX.app/Contents/MacOS/ChimeraX",
        "--script",
        str(script_path),
    ]
    assert "--nogui" not in recorded_args


def _resolve_manifest_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest_path.parent / path


def _visual_option_source(notebook_text: str) -> str:
    marker = "visual_deliverables = ["
    start = notebook_text.find(marker)
    assert start != -1
    end = notebook_text.find("return", start)
    assert end != -1
    return notebook_text[start:end]
