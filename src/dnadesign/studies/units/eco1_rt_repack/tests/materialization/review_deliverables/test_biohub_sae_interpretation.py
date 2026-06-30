"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/test_biohub_sae_interpretation.py

Biohub ESMC SAE interpretation review-deliverable tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import html
from pathlib import Path

import pyarrow.parquet as pq
import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables import (
    biohub_esmc_sae_interpretation as sae_interpretation,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables import (
    materialize_review_deliverables,
    notebook_sae_features,
    sae_structure_browser,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.fixtures import (
    write_deliverable_inputs,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.runtime_fixtures import (
    FakeMo,
)


def test_biohub_esmc_sae_interpretation_deliverables_are_rendered(tmp_path: Path) -> None:
    write_deliverable_inputs(tmp_path)

    result = materialize_review_deliverables(repo_root=Path.cwd(), output_root=tmp_path, render_chimerax_png=False)

    manifest = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))
    deliverables = {entry["deliverable_id"]: entry for entry in manifest["deliverables"]}

    activation_pattern = deliverables["biohub_esmc_wt_top_sae_feature_activation_pattern"]
    feature_heatmap = deliverables["biohub_esmc_sae_feature_activation_heatmap"]
    top_features = deliverables["biohub_esmc_protein_top_sae_features"]
    assert top_features["status"] == "materialized"
    assert activation_pattern["status"] == "rendered"
    assert feature_heatmap["status"] == "rendered"
    assert feature_heatmap["artifact_kind"] == "sae_feature_heatmap_manifest"
    top_feature_rows = pq.read_table(_resolve_manifest_path(result.manifest_path, top_features["path"])).to_pylist()
    assert {row["candidate_id"] for row in top_feature_rows} == {
        "wild_type",
        "thread_candidate_alpha",
        "thread_candidate_beta",
    }
    assert {row["selection_reason"] for row in top_feature_rows}
    assert "peak activation and prevalence" in top_features["title"]
    assert "source_backed_exact_dictionary_description" in {row["description_status"] for row in top_feature_rows}

    activation_pattern_text = _resolve_manifest_path(result.manifest_path, activation_pattern["path"]).read_text(
        encoding="utf-8"
    )
    assert "WT-active SAE features have distinct residue activation patterns" in activation_pattern_text
    assert "F101" in activation_pattern_text
    assert "activation_max" in activation_pattern["evidence_summary"]["feature_selection_rule"]
    assert "exact SAE model" not in activation_pattern["title"]
    assert "source_notebook" in activation_pattern["evidence_summary"]
    assert "esmc_sae_feature_interpretation.ipynb" in str(activation_pattern["evidence_summary"])
    assert "source-backed feature descriptions" in activation_pattern["description"]
    assert activation_pattern["evidence_summary"]["model"] == "esmc-6b-2024-12"
    assert activation_pattern["evidence_summary"]["sae_model"] == "esmc-6b-2024-12-sae-layer60-k64-codebook16384"

    heatmap_manifest = yaml.safe_load(_resolve_manifest_path(result.manifest_path, feature_heatmap["path"]).read_text())
    assert heatmap_manifest["schema_id"] == "eco1_rt.biohub_esmc_sae_feature_heatmap"
    assert heatmap_manifest["candidate_order"] == ["wild_type", "thread_candidate_alpha", "thread_candidate_beta"]
    assert heatmap_manifest["wt_sequence"] == "MKSAYL"
    assert heatmap_manifest["sequence_length"] == 6
    assert heatmap_manifest["features"][0]["feature_index"] == 101
    assert heatmap_manifest["features"][0]["label"].startswith("F101")
    assert "top tick labels are WT residue letters" in feature_heatmap["description"]
    assert "acceptance claims" in feature_heatmap["interpretation_limit"]

    loaded_heatmap = notebook_sae_features.load_sae_feature_heatmap_manifest(
        manifest_root=result.manifest_path.parent,
        selected_visual=feature_heatmap,
    )
    feature_lookup = notebook_sae_features.sae_heatmap_feature_lookup(loaded_heatmap)
    assert any(label.startswith("F101") for label in feature_lookup)
    rendered = notebook_sae_features.render_sae_feature_heatmap(
        mo=FakeMo(),
        heatmap_manifest=loaded_heatmap,
        selected_feature_index=101,
        feature_ui="<feature-dropdown>",
    )
    rendered_text = html.unescape(str(rendered))
    assert "<feature-dropdown>" in rendered_text
    assert "SAE F101" in rendered_text
    assert "WT Ec86" in rendered_text
    assert "V001" in rendered_text
    assert ">M<" in rendered_text
    assert ">6<" in rendered_text
    assert "Missing sparse entries are rendered as zero" in rendered_text
    assert "data-zoom-target" in rendered_text


def test_sae_feature_labels_stay_single_line(tmp_path: Path) -> None:
    write_deliverable_inputs(tmp_path)

    label = sae_interpretation._feature_axis_label(
        101,
        "Polymerase thumb region",
        "Fixture exact-dictionary feature description for a polymerase-like region.",
    )
    assert label == "F101 - Polymerase thumb region"
    assert "\n" not in label
    assert len(label) <= 66


def test_sae_structure_browser_descriptions_stay_concise() -> None:
    description = (
        "Summary: Right-hand nucleic-acid polymerase module, with a strong preference for the C-terminal "
        "helical thumb/CTE and adjacent palm region that contacts and positions the template-product duplex. "
        "Activation pattern: many long source details that belong in the SAE feature inspector, not the "
        "structure-browser title region."
    )

    concise = sae_structure_browser._concise_sae_description(description)

    assert concise.startswith("Right-hand nucleic-acid polymerase module")
    assert "Activation pattern" not in concise
    assert len(concise) <= 261


def _resolve_manifest_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest_path.parent / path
