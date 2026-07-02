"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/biohub_esmc_sequence_pseudolikelihood/test_materialization.py

Eco1 Biohub ESMC sequence pseudo-likelihood materialization tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest
import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.biohub_esmc_sequence_pseudolikelihood import (
    materialize_biohub_esmc_sequence_pseudolikelihood,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.atlas_semantic_profile._fixtures import (
    write_foldcheck_report_fixture,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.biohub_esmc_wt_mutation_scoring._fixtures import (
    FakeSequenceLogitsClient,
)


def test_sequence_pseudolikelihood_materializes_two_sequence_capped_smoke(tmp_path: Path) -> None:
    write_foldcheck_report_fixture(tmp_path, accepted_candidate_ids={"wild_type", "thread_candidate_test"})

    result = materialize_biohub_esmc_sequence_pseudolikelihood(
        repo_root=Path.cwd(),
        output_root=tmp_path,
        positions="all",
        sequence_limit="all",
        max_new_requests=8,
        biohub_client=FakeSequenceLogitsClient(),
        retrieved_at="2026-07-02T00:00:00Z",
    )

    position_rows = pq.read_table(result.position_pll_path).to_pylist()
    sequence_rows = pq.read_table(result.sequence_pll_path).to_pylist()
    by_id = {row["sequence_id"]: row for row in sequence_rows}
    assert result.selected_sequence_count == 2
    assert result.selected_position_count == 320
    assert len(position_rows) == 640
    assert sum(1 for row in position_rows if row["status"] == "accepted") == 8
    assert by_id["wild_type"]["status"] == "partial"
    assert by_id["wild_type"]["delta_pll_total_vs_wt"] is None
    assert by_id["wild_type"]["delta_pll_mean_vs_wt"] is None
    assert by_id["thread_candidate_test"]["status"] == "partial"
    manifest_text = result.request_manifest_path.read_text(encoding="utf-8")
    assert "fixture-secret" not in manifest_text
    assert "authorization: <redacted>" in manifest_text
    manifest = yaml.safe_load(manifest_text)
    assert manifest["endpoint_flow"] == ["POST /api/v1/encode", "POST /api/v1/logits"]
    assert manifest["materialization_status"] == "partial"
    assert manifest["scoring_method_id"] == "esmc_leave_one_out_pseudolikelihood_v1"


def test_sequence_pseudolikelihood_capped_run_keeps_partial_rows(tmp_path: Path) -> None:
    write_foldcheck_report_fixture(tmp_path, accepted_candidate_ids={"wild_type", "thread_candidate_test"})

    result = materialize_biohub_esmc_sequence_pseudolikelihood(
        repo_root=Path.cwd(),
        output_root=tmp_path,
        positions="all",
        sequence_limit="all",
        max_new_requests=1,
        biohub_client=FakeSequenceLogitsClient(),
        retrieved_at="2026-07-02T00:00:00Z",
    )

    sequence_rows = pq.read_table(result.sequence_pll_path).to_pylist()
    assert {row["status"] for row in sequence_rows} == {"partial"}
    manifest = yaml.safe_load(result.request_manifest_path.read_text(encoding="utf-8"))
    assert manifest["materialization_mode"] == "resumable_capped"
    assert manifest["materialization_status"] == "partial"


def test_sequence_pseudolikelihood_rejects_partial_uncapped_final_run(tmp_path: Path) -> None:
    write_foldcheck_report_fixture(tmp_path, accepted_candidate_ids={"wild_type", "thread_candidate_test"})

    with pytest.raises(ValueError, match="requires all positions"):
        materialize_biohub_esmc_sequence_pseudolikelihood(
            repo_root=Path.cwd(),
            output_root=tmp_path,
            positions="1-2",
            sequence_limit="all",
            biohub_client=FakeSequenceLogitsClient(),
            retrieved_at="2026-07-02T00:00:00Z",
        )


def test_sequence_pseudolikelihood_candidate_filter_keeps_wt_control(tmp_path: Path) -> None:
    write_foldcheck_report_fixture(tmp_path, accepted_candidate_ids={"wild_type", "thread_candidate_test"})

    result = materialize_biohub_esmc_sequence_pseudolikelihood(
        repo_root=Path.cwd(),
        output_root=tmp_path,
        positions="1",
        sequence_limit="all",
        candidate_ids=("thread_candidate_test",),
        max_new_requests=2,
        biohub_client=FakeSequenceLogitsClient(),
        retrieved_at="2026-07-02T00:00:00Z",
    )

    position_rows = pq.read_table(result.position_pll_path).to_pylist()
    assert {row["sequence_id"] for row in position_rows} == {"wild_type", "thread_candidate_test"}
