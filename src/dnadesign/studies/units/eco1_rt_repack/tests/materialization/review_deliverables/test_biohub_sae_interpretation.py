"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/test_biohub_sae_interpretation.py

Biohub ESMC SAE interpretation review-deliverable tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables import (
    biohub_esmc_sae_fold_llr as sae_fold_llr,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables import (
    biohub_esmc_sae_interpretation as sae_interpretation,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables import (
    materialize_review_deliverables,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.fixtures import (
    write_deliverable_inputs,
)


def test_biohub_esmc_sae_interpretation_deliverables_are_rendered(tmp_path: Path) -> None:
    write_deliverable_inputs(tmp_path)

    result = materialize_review_deliverables(repo_root=Path.cwd(), output_root=tmp_path, render_chimerax_png=False)

    manifest = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))
    deliverables = {entry["deliverable_id"]: entry for entry in manifest["deliverables"]}

    activation_pattern = deliverables["biohub_esmc_wt_top_sae_feature_activation_pattern"]
    activation_ratio = deliverables["biohub_esmc_candidate_top_sae_feature_activation_ratio"]
    top_features = deliverables["biohub_esmc_protein_top_sae_features"]
    fold_llr_comparison = deliverables["biohub_esmc_sae_fold_llr_comparison"]
    assert top_features["status"] == "materialized"
    assert activation_pattern["status"] == "rendered"
    assert activation_ratio["status"] == "rendered"
    assert fold_llr_comparison["status"] == "rendered"
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

    activation_ratio_text = _resolve_manifest_path(result.manifest_path, activation_ratio["path"]).read_text(
        encoding="utf-8"
    )
    assert "Candidates vary in WT-active SAE activation ratios" in activation_ratio_text
    assert "candidate activation sum / WT activation sum" in activation_ratio_text
    assert "acceptance claims" in activation_ratio["interpretation_limit"]

    fold_llr_text = _resolve_manifest_path(result.manifest_path, fold_llr_comparison["path"]).read_text(
        encoding="utf-8"
    )
    assert "SAE similarity, ColabFold confidence, and ESMC mutation scores are compared together" in fold_llr_text
    assert "WT Ec86" in fold_llr_text
    assert "V001" in fold_llr_text
    assert "Fixture exact-dictionary feature description" in fold_llr_text
    assert "ProteinMPNN variant ordered by SAE similarity" in fold_llr_text
    assert "Feature rows compare WT-normalized activation" in fold_llr_text
    assert "pLDDT" in fold_llr_text
    assert "single-substitution LLR" in fold_llr_text
    svg_desc = re.search(r"<desc[^>]*>(.*?)</desc>", fold_llr_text, flags=re.DOTALL)
    assert svg_desc is not None
    assert len(svg_desc.group(1)) < 1200
    assert "SAE similarity to WT" in fold_llr_comparison["description"]
    assert "rank variants" not in fold_llr_text
    assert "activation patterns are compared" in fold_llr_text
    assert "LLR sum, scaled within panel" in fold_llr_text
    assert "not a joint protein likelihood" in fold_llr_comparison["interpretation_limit"]
    assert fold_llr_comparison["evidence_summary"]["sequence_rows"] == 3
    assert fold_llr_comparison["evidence_summary"]["llr_scoring_rule"] == "sum_variant_single_substitution_llrs"


def test_sae_feature_labels_stay_single_line(tmp_path: Path) -> None:
    write_deliverable_inputs(tmp_path)

    feature_catalog_path = tmp_path / "biohub_esmc_feature_catalog.parquet"
    label = sae_interpretation._feature_axis_label(
        101,
        "",
        "Fixture exact-dictionary feature description for a polymerase-like region.",
    )
    assert label.startswith("F101 - Fixture exact-dictionary feature description")
    assert "\n" not in label
    assert len(label) <= 66
    assert all("\n" not in label for label in sae_fold_llr._feature_labels(feature_catalog_path, [101, 202]))


def test_sae_fold_llr_svg_description_stays_concise() -> None:
    description = sae_fold_llr._panel_accessibility_description(
        {
            "feature_labels": ["F101 - " + "long description " * 200] * 12,
            "row_labels": ["WT Ec86"] + [f"V{index:03d}" for index in range(1, 97)],
            "feature_descriptions": ["full source-backed description " * 500],
        }
    )

    assert len(description) < 500
    assert "full source-backed description" not in description
    assert "feature inspector" in description


def test_sae_fold_llr_rejects_malformed_candidate_mutations(tmp_path: Path) -> None:
    candidate_table = tmp_path / "candidate_table.parquet"
    wt_llr = tmp_path / "wt_substitution_llr.parquet"
    pq.write_table(
        pa.Table.from_pylist([{"candidate_id": "thread_candidate_bad", "canonical_mutations": ["bad"]}]),
        candidate_table,
    )
    pq.write_table(
        pa.Table.from_pylist([{"canonical_position": 1, "alt_aa": "G", "llr": -1.0}]),
        wt_llr,
    )

    with pytest.raises(ValueError, match="Malformed canonical mutation"):
        sae_fold_llr._llr_sum_by_candidate(candidate_table, wt_llr)


def test_sae_fold_llr_rejects_missing_substitution_scores(tmp_path: Path) -> None:
    candidate_table = tmp_path / "candidate_table.parquet"
    wt_llr = tmp_path / "wt_substitution_llr.parquet"
    pq.write_table(
        pa.Table.from_pylist([{"candidate_id": "thread_candidate_bad", "canonical_mutations": ["A1G"]}]),
        candidate_table,
    )
    pq.write_table(
        pa.Table.from_pylist([{"canonical_position": 1, "alt_aa": "V", "llr": -1.0}]),
        wt_llr,
    )

    with pytest.raises(ValueError, match="Missing ESMC LLR"):
        sae_fold_llr._llr_sum_by_candidate(candidate_table, wt_llr)


def _resolve_manifest_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest_path.parent / path
