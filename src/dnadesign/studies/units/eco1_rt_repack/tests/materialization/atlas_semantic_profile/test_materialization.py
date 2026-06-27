"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/atlas_semantic_profile/test_materialization.py

Eco1 ESM Atlas semantic-profile materialization tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest
import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.atlas_semantic_profile import (
    materialize_atlas_semantic_profile,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.atlas_semantic_profile.selection import (
    select_fold_accepted_atlas_sequences,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.atlas_semantic_profile._fixtures import (
    FakeAtlasClient,
    write_foldcheck_report_fixture,
)


def test_atlas_semantic_profile_materializes_wt_only_smoke(tmp_path: Path) -> None:
    request_manifest = write_foldcheck_report_fixture(tmp_path, accepted_candidate_ids={"wild_type"})

    result = materialize_atlas_semantic_profile(
        repo_root=Path.cwd(),
        output_root=tmp_path,
        sequence_limit="1",
        atlas_client=FakeAtlasClient(),
        retrieved_at="2026-06-25T00:00:00Z",
    )

    profile_rows = pq.read_table(result.profile_path).to_pylist()
    assert [row["candidate_id"] for row in profile_rows] == ["wild_type"]
    assert profile_rows[0]["source_request_hash"] == str(request_manifest["request_hash"])
    assert profile_rows[0]["status"] == "accepted"
    assert pq.read_table(result.protein_activations_path).num_rows == 2
    assert pq.read_table(result.residue_activations_path).num_rows == 4
    assert pq.read_table(result.feature_catalog_path).num_rows == 2
    assert pq.read_table(result.structure_prediction_registry_path).num_rows == 0


def test_atlas_semantic_profile_records_api_failures_without_on_demand_folding(tmp_path: Path) -> None:
    write_foldcheck_report_fixture(tmp_path, accepted_candidate_ids={"wild_type", "thread_candidate_test"})
    fake_client = FakeAtlasClient(fail_for={"thread_candidate_test"})

    result = materialize_atlas_semantic_profile(
        repo_root=Path.cwd(),
        output_root=tmp_path,
        sequence_limit="all",
        atlas_client=fake_client,
        retrieved_at="2026-06-25T00:00:00Z",
    )

    profile_rows = pq.read_table(result.profile_path).to_pylist()
    statuses = {row["candidate_id"]: row["status"] for row in profile_rows}
    assert statuses == {"wild_type": "accepted", "thread_candidate_test": "errored"}
    errored = next(row for row in profile_rows if row["candidate_id"] == "thread_candidate_test")
    assert "Atlas API fixture miss" in errored["failure_reason"]
    assert fake_client.fold_on_miss_values == [False, False]


def test_atlas_semantic_profile_uses_declared_subset_manifest(tmp_path: Path) -> None:
    request_manifest = write_foldcheck_report_fixture(
        tmp_path, accepted_candidate_ids={"wild_type", "thread_candidate_test"}
    )
    manifest_path = tmp_path / "atlas_subset_manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "schema_id": "eco1_rt.atlas_subset_manifest",
                "source_request_hash": request_manifest["request_hash"],
                "selected_sequence_ids": ["thread_candidate_test"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    selection = select_fold_accepted_atlas_sequences(
        output_root=tmp_path,
        sequence_limit="all",
        selection_manifest_path=manifest_path,
    )

    assert [record.sequence_id for record in selection.records] == ["thread_candidate_test"]


def test_atlas_semantic_profile_rejects_unaccepted_subset_manifest_ids(tmp_path: Path) -> None:
    write_foldcheck_report_fixture(tmp_path, accepted_candidate_ids={"wild_type"})
    manifest_path = tmp_path / "atlas_subset_manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "schema_id": "eco1_rt.atlas_subset_manifest",
                "selected_sequence_ids": ["thread_candidate_test"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not fold-accepted"):
        select_fold_accepted_atlas_sequences(
            output_root=tmp_path,
            sequence_limit="all",
            selection_manifest_path=manifest_path,
        )


def test_atlas_on_demand_folding_requires_prediction_set_id(tmp_path: Path) -> None:
    write_foldcheck_report_fixture(tmp_path, accepted_candidate_ids={"wild_type"})

    with pytest.raises(ValueError, match="prediction_set_id"):
        materialize_atlas_semantic_profile(
            repo_root=Path.cwd(),
            output_root=tmp_path,
            sequence_limit="1",
            atlas_client=FakeAtlasClient(),
            allow_fold_on_miss=True,
            retrieved_at="2026-06-25T00:00:00Z",
        )


def test_atlas_on_demand_folding_writes_separate_structure_registry(tmp_path: Path) -> None:
    write_foldcheck_report_fixture(tmp_path, accepted_candidate_ids={"wild_type", "thread_candidate_test"})

    result = materialize_atlas_semantic_profile(
        repo_root=Path.cwd(),
        output_root=tmp_path,
        sequence_limit="all",
        atlas_client=FakeAtlasClient(folded_for={"thread_candidate_test"}),
        allow_fold_on_miss=True,
        prediction_set_id="atlas_fixture_fold_on_miss",
        retrieved_at="2026-06-25T00:00:00Z",
    )

    profile_rows = pq.read_table(result.profile_path).to_pylist()
    folded_by_candidate = {row["candidate_id"]: row["folded_on_demand"] for row in profile_rows}
    registry_rows = pq.read_table(result.structure_prediction_registry_path).to_pylist()

    assert folded_by_candidate == {"wild_type": False, "thread_candidate_test": True}
    assert len(registry_rows) == 1
    assert registry_rows[0]["candidate_id"] == "thread_candidate_test"
    assert registry_rows[0]["backend_kind"] == "esm_atlas"
    assert Path(registry_rows[0]["local_structure_path"]).exists()


def test_atlas_selection_requires_fold_accepted_rows(tmp_path: Path) -> None:
    write_foldcheck_report_fixture(tmp_path, accepted_candidate_ids={"wild_type"})

    selection = select_fold_accepted_atlas_sequences(output_root=tmp_path, sequence_limit="all")

    assert [record.sequence_id for record in selection.records] == ["wild_type"]
