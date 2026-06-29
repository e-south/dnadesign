"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/biohub_esmc_wt_mutation_scoring/test_materialization.py

Eco1 WT-only Biohub ESMC mutation-scoring materialization tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.biohub_esmc_wt_mutation_scoring import (
    materialize_biohub_esmc_wt_mutation_scoring,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.atlas_semantic_profile._fixtures import (
    write_foldcheck_report_fixture,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.biohub_esmc_wt_mutation_scoring._fixtures import (
    FakeSequenceLogitsClient,
    rewrite_position_table_with_null_alternate_fraction,
    rewrite_position_table_with_old_fraction_name,
    write_mask_set,
)


def test_wt_mutation_scoring_materializes_two_position_smoke(tmp_path: Path) -> None:
    write_foldcheck_report_fixture(tmp_path, accepted_candidate_ids={"wild_type"})
    write_mask_set(tmp_path / "mask_set.yaml", length=4)

    result = materialize_biohub_esmc_wt_mutation_scoring(
        repo_root=Path.cwd(),
        output_root=tmp_path,
        positions="1-2",
        biohub_client=FakeSequenceLogitsClient(),
        retrieved_at="2026-06-27T00:00:00Z",
    )

    position_rows = pq.read_table(result.position_entropy_path).to_pylist()
    substitution_rows = pq.read_table(result.substitution_llr_path).to_pylist()
    mask_join_rows = pq.read_table(result.mask_join_path).to_pylist()
    assert len(position_rows) == 2
    assert len(substitution_rows) == 38
    assert len(mask_join_rows) == 2
    assert {row["sequence_id"] for row in position_rows} == {"wild_type"}
    assert position_rows[0]["status"] == "accepted"
    assert mask_join_rows[0]["mask_context_status"] == "joined"
    manifest_text = result.request_manifest_path.read_text(encoding="utf-8")
    assert "fixture-secret" not in manifest_text
    assert "authorization: <redacted>" in manifest_text
    assert "changes_current_mask: false" in manifest_text
    manifest = yaml.safe_load(manifest_text)
    method_urls = {reference["url"] for reference in manifest["method_references"]}
    notebook_url = (
        "https://colab.research.google.com/github/Biohub/esm/blob/main/cookbook/tutorials/esmc_mutation_scoring.ipynb"
    )
    assert notebook_url in method_urls
    assert "https://www.biohub.ai/api-reference/logits" in method_urls
    entropy_plot_text = (result.plots_root / "wt_entropy_by_position.svg").read_text(encoding="utf-8")
    assert "RT1-RT7 review spans" in entropy_plot_text
    assert "Protected residues" in entropy_plot_text
    assert "Motif anchors" in entropy_plot_text


def test_wt_mutation_scoring_max_new_requests_writes_resumable_error(tmp_path: Path) -> None:
    write_foldcheck_report_fixture(tmp_path, accepted_candidate_ids={"wild_type"})
    write_mask_set(tmp_path / "mask_set.yaml", length=4)

    result = materialize_biohub_esmc_wt_mutation_scoring(
        repo_root=Path.cwd(),
        output_root=tmp_path,
        positions="1-2",
        max_new_requests=1,
        biohub_client=FakeSequenceLogitsClient(),
        retrieved_at="2026-06-27T00:00:00Z",
    )

    position_rows = pq.read_table(result.position_entropy_path).to_pylist()
    substitution_rows = pq.read_table(result.substitution_llr_path).to_pylist()
    statuses = {row["canonical_position"]: row["status"] for row in position_rows}
    failure_reasons = {row["canonical_position"]: row["failure_reason"] for row in position_rows}
    assert statuses == {1: "accepted", 2: "errored"}
    assert failure_reasons[2] == "biohub_request_not_attempted_due_to_max_new_requests"
    assert len(substitution_rows) == 19


def test_wt_mutation_scoring_resume_ignores_stale_position_schema(tmp_path: Path) -> None:
    write_foldcheck_report_fixture(tmp_path, accepted_candidate_ids={"wild_type"})
    write_mask_set(tmp_path / "mask_set.yaml", length=4)

    first = materialize_biohub_esmc_wt_mutation_scoring(
        repo_root=Path.cwd(),
        output_root=tmp_path,
        positions="1",
        biohub_client=FakeSequenceLogitsClient(),
        retrieved_at="2026-06-27T00:00:00Z",
    )
    rewrite_position_table_with_old_fraction_name(first.position_entropy_path)

    second = materialize_biohub_esmc_wt_mutation_scoring(
        repo_root=Path.cwd(),
        output_root=tmp_path,
        positions="1",
        resume_existing=True,
        max_new_requests=0,
        biohub_client=FakeSequenceLogitsClient(),
        retrieved_at="2026-06-27T00:00:00Z",
    )

    position_rows = pq.read_table(second.position_entropy_path).to_pylist()
    assert position_rows[0]["status"] == "errored"
    assert position_rows[0]["failure_reason"] == "biohub_request_not_attempted_due_to_max_new_requests"


def test_wt_mutation_scoring_resume_ignores_null_accepted_metric(tmp_path: Path) -> None:
    write_foldcheck_report_fixture(tmp_path, accepted_candidate_ids={"wild_type"})
    write_mask_set(tmp_path / "mask_set.yaml", length=4)

    first = materialize_biohub_esmc_wt_mutation_scoring(
        repo_root=Path.cwd(),
        output_root=tmp_path,
        positions="1",
        biohub_client=FakeSequenceLogitsClient(),
        retrieved_at="2026-06-27T00:00:00Z",
    )
    rewrite_position_table_with_null_alternate_fraction(first.position_entropy_path)

    second = materialize_biohub_esmc_wt_mutation_scoring(
        repo_root=Path.cwd(),
        output_root=tmp_path,
        positions="1",
        resume_existing=True,
        max_new_requests=0,
        biohub_client=FakeSequenceLogitsClient(),
        retrieved_at="2026-06-27T00:00:00Z",
    )

    position_rows = pq.read_table(second.position_entropy_path).to_pylist()
    assert position_rows[0]["status"] == "errored"
    assert position_rows[0]["failure_reason"] == "biohub_request_not_attempted_due_to_max_new_requests"
