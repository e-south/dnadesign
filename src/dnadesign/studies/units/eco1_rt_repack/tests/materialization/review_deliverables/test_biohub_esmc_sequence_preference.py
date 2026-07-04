"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/test_biohub_esmc_sequence_preference.py

Biohub ESMC candidate-preference review-deliverable tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables import (
    biohub_esmc_sae_umap,
    materialize_review_deliverables,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables import (
    biohub_esmc_sequence_preference as sequence_preference,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.esmc_fixtures import (
    write_wt_mutation_scoring_outputs,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.fixtures import (
    write_deliverable_inputs,
)


@pytest.fixture(autouse=True)
def _fast_sae_embedding_for_sequence_preference_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep these ESMC tests focused; SAE-specific tests cover real UMAP embedding."""

    def _linear_embedding_for_test(matrix):
        return biohub_esmc_sae_umap._linear_embedding(matrix), "linear_test_embedding"

    monkeypatch.setattr(biohub_esmc_sae_umap, "_embed_delta_matrix", _linear_embedding_for_test)


def test_biohub_esmc_sequence_preference_deliverables_are_rendered(tmp_path: Path) -> None:
    write_deliverable_inputs(tmp_path)

    result = materialize_review_deliverables(repo_root=Path.cwd(), output_root=tmp_path, render_chimerax_png=False)

    manifest = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))
    deliverables = {entry["deliverable_id"]: entry for entry in manifest["deliverables"]}
    scoring_manifest_row = deliverables["biohub_esmc_sequence_scoring_manifest"]
    table_row = deliverables["biohub_esmc_variant_llr_scores"]
    plot_row = deliverables["biohub_esmc_candidate_preference_vs_wt"]

    assert scoring_manifest_row["status"] == "materialized"
    assert table_row["status"] == "materialized"
    assert plot_row["status"] == "rendered"
    assert "not a whole-protein pseudo-likelihood" in plot_row["interpretation_limit"]
    assert plot_row["evidence_summary"]["candidate_count"] == 2
    assert plot_row["evidence_summary"]["scoring_method_id"] == "esmc_additive_wt_single_substitution_llr_v1"

    scoring_manifest = yaml.safe_load(
        _resolve_manifest_path(result.manifest_path, scoring_manifest_row["path"]).read_text(encoding="utf-8")
    )
    assert scoring_manifest["schema_id"] == "eco1_rt.biohub_esmc.sequence_scoring_manifest"
    assert scoring_manifest["candidate_count"] == 2
    assert scoring_manifest["additional_biohub_request_count"] == 0
    assert scoring_manifest["model"] == "esmc-300m-2024-12"
    assert scoring_manifest["biohub_api_base_url"] == "https://biohub.ai"
    assert scoring_manifest["endpoint_flow"] == ["POST /api/v1/encode", "POST /api/v1/logits"]
    assert scoring_manifest["source_scoring_method_id"] == "esmc_masked_marginal_v1"
    assert scoring_manifest["scoring_method_id"] == "esmc_additive_wt_single_substitution_llr_v1"
    assert scoring_manifest["whole_protein_pseudolikelihood_status"] == "not_materialized_request_heavy"
    assert scoring_manifest["authorization"] == "<redacted>"

    score_path = _resolve_manifest_path(result.manifest_path, table_row["path"])
    score_rows = pq.read_table(score_path).to_pylist()
    assert [row["candidate_id"] for row in score_rows] == ["thread_candidate_alpha", "thread_candidate_beta"]
    assert all(row["status"] == "accepted" for row in score_rows)
    assert all(row["mutation_count"] > 0 for row in score_rows)
    assert all(row["llr_per_mutation"] is not None for row in score_rows)
    assert score_rows[0]["model"] == "esmc-300m-2024-12"
    assert score_rows[0]["scoring_method_id"] == "esmc_additive_wt_single_substitution_llr_v1"

    plot_text = _resolve_manifest_path(result.manifest_path, plot_row["path"]).read_text(encoding="utf-8")
    assert "Candidate ESMC additive LLR versus wild type" in plot_text
    assert "WT-context single-substitution LLR sum" in plot_text
    assert "thread_candidate_alpha" not in plot_text
    assert "Activity" not in plot_text


def test_biohub_esmc_sequence_preference_adds_6b_lane_and_model_agreement(tmp_path: Path) -> None:
    write_deliverable_inputs(tmp_path)
    write_wt_mutation_scoring_outputs(
        tmp_path,
        scoring_root=tmp_path / "biohub_esmc" / "mutation_scoring" / "esmc_6b_2024_12",
        model="esmc-6b-2024-12",
        request_hash_tail="8",
        llr_shift=0.2,
    )
    stale_plot_path = (
        tmp_path
        / "review_deliverables"
        / "biohub_esmc_sequence_scoring"
        / "esmc_candidate_preference_model_stability.svg"
    )
    stale_plot_path.parent.mkdir(parents=True, exist_ok=True)
    stale_plot_path.write_text("<svg><title>stale model stability</title></svg>", encoding="utf-8")

    result = materialize_review_deliverables(repo_root=Path.cwd(), output_root=tmp_path, render_chimerax_png=False)

    manifest = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))
    deliverables = {entry["deliverable_id"]: entry for entry in manifest["deliverables"]}
    assert deliverables["biohub_esmc_6b_sequence_scoring_manifest"]["status"] == "materialized"
    assert deliverables["biohub_esmc_6b_variant_llr_scores"]["status"] == "materialized"
    assert deliverables["biohub_esmc_6b_candidate_preference_vs_wt"]["status"] == "rendered"
    assert deliverables["biohub_esmc_candidate_preference_model_agreement"]["status"] == "rendered"
    assert "biohub_esmc_candidate_preference_model_stability" not in deliverables

    six_b_manifest = yaml.safe_load(
        _resolve_manifest_path(
            result.manifest_path,
            deliverables["biohub_esmc_6b_sequence_scoring_manifest"]["path"],
        ).read_text(encoding="utf-8")
    )
    assert six_b_manifest["model"] == "esmc-6b-2024-12"
    assert six_b_manifest["scoring_method_id"] == "esmc_6b_2024_12_additive_wt_single_substitution_llr_v1"

    agreement_row = deliverables["biohub_esmc_candidate_preference_model_agreement"]
    assert agreement_row["role"] == "review_only"
    assert agreement_row["evidence_summary"]["candidate_count"] == 2
    assert agreement_row["evidence_summary"]["left_model"] == "esmc-300m-2024-12"
    assert agreement_row["evidence_summary"]["right_model"] == "esmc-6b-2024-12"
    agreement_plot_text = _resolve_manifest_path(result.manifest_path, agreement_row["path"]).read_text(
        encoding="utf-8"
    )
    assert "300M and 6B ESMC additive LLR scores disagree" in agreement_plot_text
    assert "stability" not in agreement_row["title"].lower()
    assert not stale_plot_path.exists()


def test_sequence_preference_rejects_malformed_candidate_mutations(tmp_path: Path) -> None:
    candidate_table = tmp_path / "candidate_table.parquet"
    wt_llr = tmp_path / "wt_substitution_llr.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [{"candidate_id": "thread_candidate_bad", "canonical_mutations": ["bad"], "mutation_count": 1}]
        ),
        candidate_table,
    )
    pq.write_table(
        pa.Table.from_pylist([{"canonical_position": 1, "alt_aa": "G", "llr": -1.0}]),
        wt_llr,
    )

    with pytest.raises(ValueError, match="Malformed canonical mutation"):
        sequence_preference.build_variant_llr_score_rows(
            candidate_table_path=candidate_table,
            wt_substitution_llr_path=wt_llr,
            wt_mutation_scoring_manifest_path=None,
            foldcheck_ranking_path=None,
        )


def test_sequence_preference_rejects_missing_substitution_scores(tmp_path: Path) -> None:
    candidate_table = tmp_path / "candidate_table.parquet"
    wt_llr = tmp_path / "wt_substitution_llr.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [{"candidate_id": "thread_candidate_bad", "canonical_mutations": ["A1G"], "mutation_count": 1}]
        ),
        candidate_table,
    )
    pq.write_table(
        pa.Table.from_pylist([{"canonical_position": 1, "alt_aa": "V", "llr": -1.0}]),
        wt_llr,
    )

    with pytest.raises(ValueError, match="Missing ESMC LLR"):
        sequence_preference.build_variant_llr_score_rows(
            candidate_table_path=candidate_table,
            wt_substitution_llr_path=wt_llr,
            wt_mutation_scoring_manifest_path=None,
            foldcheck_ranking_path=None,
        )


def _resolve_manifest_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest_path.parent / path
