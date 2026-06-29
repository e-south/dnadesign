"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/test_biohub_sae_interpretation.py

Biohub ESMC SAE interpretation review-deliverable tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
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

    localization = deliverables["biohub_esmc_wt_top_sae_feature_localization"]
    retention = deliverables["biohub_esmc_candidate_top_sae_feature_retention"]
    top_features = deliverables["biohub_esmc_protein_top_sae_features"]
    fold_llr_comparison = deliverables["biohub_esmc_sae_fold_llr_comparison"]
    assert top_features["status"] == "materialized"
    assert localization["status"] == "rendered"
    assert retention["status"] == "rendered"
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

    localization_text = _resolve_manifest_path(result.manifest_path, localization["path"]).read_text(encoding="utf-8")
    assert "WT-active SAE features localize to specific Ec86 regions" in localization_text
    assert "F101" in localization_text
    assert "activation_max" in localization["evidence_summary"]["feature_selection_rule"]
    assert "exact SAE model" not in localization["title"]
    assert "source_notebook" in localization["evidence_summary"]
    assert "esmc_sae_feature_interpretation.ipynb" in str(localization["evidence_summary"])
    assert "source-backed feature descriptions" in localization["description"]

    retention_text = _resolve_manifest_path(result.manifest_path, retention["path"]).read_text(encoding="utf-8")
    assert "Candidates retain or shift WT-active SAE features" in retention_text
    assert "candidate activation sum / WT activation sum" in retention_text
    assert "acceptance claims" in retention["interpretation_limit"]

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
    assert "ESMC single-substitution LLR sum" in fold_llr_text
    assert "SAE similarity rank" in fold_llr_comparison["description"]
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


def _resolve_manifest_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest_path.parent / path
