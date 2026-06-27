"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/atlas_semantic_profile/test_resume.py

Resume tests for Eco1 ESM Atlas semantic-profile materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.atlas_semantic_profile import (
    materialize_atlas_semantic_profile,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.atlas_semantic_profile._fixtures import (
    FakeAtlasClient,
    write_foldcheck_report_fixture,
)


def test_atlas_semantic_profile_resumes_existing_rows(tmp_path: Path) -> None:
    write_foldcheck_report_fixture(tmp_path, accepted_candidate_ids={"wild_type", "thread_candidate_test"})
    materialize_atlas_semantic_profile(
        repo_root=Path.cwd(),
        output_root=tmp_path,
        sequence_limit="1",
        atlas_client=FakeAtlasClient(),
        retrieved_at="2026-06-25T00:00:00Z",
    )
    resume_client = FakeAtlasClient()

    result = materialize_atlas_semantic_profile(
        repo_root=Path.cwd(),
        output_root=tmp_path,
        sequence_limit="all",
        atlas_client=resume_client,
        retrieved_at="2026-06-25T00:05:00Z",
        resume_existing=True,
    )

    profile_rows = pq.read_table(result.profile_path).to_pylist()
    assert {row["candidate_id"] for row in profile_rows} == {"wild_type", "thread_candidate_test"}
    assert [row["retrieved_at"] for row in profile_rows if row["candidate_id"] == "wild_type"] == [
        "2026-06-25T00:00:00Z"
    ]
    assert resume_client.fold_on_miss_values == [False]


def test_atlas_on_demand_resume_preserves_structure_registry(tmp_path: Path) -> None:
    write_foldcheck_report_fixture(tmp_path, accepted_candidate_ids={"wild_type", "thread_candidate_test"})
    materialize_atlas_semantic_profile(
        repo_root=Path.cwd(),
        output_root=tmp_path,
        sequence_limit="all",
        atlas_client=FakeAtlasClient(folded_for={"thread_candidate_test"}),
        allow_fold_on_miss=True,
        prediction_set_id="atlas_fixture_fold_on_miss",
        retrieved_at="2026-06-25T00:00:00Z",
    )
    resume_client = FakeAtlasClient(fail_for={"wild_type", "thread_candidate_test"})

    result = materialize_atlas_semantic_profile(
        repo_root=Path.cwd(),
        output_root=tmp_path,
        sequence_limit="all",
        atlas_client=resume_client,
        allow_fold_on_miss=True,
        prediction_set_id="atlas_fixture_fold_on_miss",
        resume_existing=True,
        max_new_requests=0,
        retrieved_at="2026-06-25T00:05:00Z",
    )

    profile_rows = pq.read_table(result.profile_path).to_pylist()
    registry_rows = pq.read_table(result.structure_prediction_registry_path).to_pylist()

    assert {row["candidate_id"] for row in profile_rows} == {"wild_type", "thread_candidate_test"}
    assert [row["retrieved_at"] for row in profile_rows if row["candidate_id"] == "thread_candidate_test"] == [
        "2026-06-25T00:00:00Z"
    ]
    assert len(registry_rows) == 1
    assert registry_rows[0]["candidate_id"] == "thread_candidate_test"
    assert Path(registry_rows[0]["local_structure_path"]).exists()
    assert resume_client.fold_on_miss_values == []
