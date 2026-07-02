"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/biohub_esmc_sae_profile/test_materialization.py

Eco1 Biohub ESMC SAE-profile materialization tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.biohub_esmc_sae_profile import (
    materialize_biohub_esmc_sae_profile,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.atlas_semantic_profile._fixtures import (
    write_foldcheck_report_fixture,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.biohub_esmc_sae_profile.fixtures import (
    FakeBiohubEsmcClient,
    MalformedBiohubEsmcClient,
    TimeoutOnceBiohubEsmcClient,
)

_FIXTURE_SAE_MODEL = "fixture-sae-model"


def test_biohub_esmc_sae_profile_materializes_wt_only_smoke(tmp_path: Path) -> None:
    write_foldcheck_report_fixture(tmp_path, accepted_candidate_ids={"wild_type"})

    result = materialize_biohub_esmc_sae_profile(
        repo_root=Path.cwd(),
        output_root=tmp_path,
        sequence_limit="1",
        sae_model=_FIXTURE_SAE_MODEL,
        biohub_client=FakeBiohubEsmcClient(),
        retrieved_at="2026-06-25T00:00:00Z",
    )

    profile_rows = pq.read_table(result.profile_path).to_pylist()
    assert [row["candidate_id"] for row in profile_rows] == ["wild_type"]
    assert profile_rows[0]["status"] == "accepted"
    assert profile_rows[0]["key_label"] == "bu-dunlop-lab"
    assert pq.read_table(result.protein_features_path).num_rows == 2
    assert pq.read_table(result.residue_features_path).num_rows == 3
    manifest_text = result.request_manifest_path.read_text(encoding="utf-8")
    assert "fixture-secret" not in manifest_text
    assert "authorization: <redacted>" in manifest_text
    assert "esmc_sae_feature_interpretation.ipynb" in manifest_text
    assert "https://www.biohub.ai/api-reference/logits" in manifest_text
    assert "https://huggingface.co/biohub/ESMC-6B-sae-layer60-k64-codebook16384" in manifest_text


def test_biohub_esmc_sae_profile_caps_new_requests_with_explicit_rows(tmp_path: Path) -> None:
    write_foldcheck_report_fixture(tmp_path, accepted_candidate_ids={"wild_type", "thread_candidate_test"})

    result = materialize_biohub_esmc_sae_profile(
        repo_root=Path.cwd(),
        output_root=tmp_path,
        sequence_limit="all",
        max_new_requests=1,
        sae_model=_FIXTURE_SAE_MODEL,
        biohub_client=FakeBiohubEsmcClient(),
        retrieved_at="2026-06-25T00:00:00Z",
    )

    profile_rows = pq.read_table(result.profile_path).to_pylist()
    statuses = {row["candidate_id"]: row["status"] for row in profile_rows}
    failure_reasons = {row["candidate_id"]: row["failure_reason"] for row in profile_rows}
    assert statuses == {"wild_type": "accepted", "thread_candidate_test": "errored"}
    assert failure_reasons["thread_candidate_test"] == "biohub_request_not_attempted_due_to_max_new_requests"


def test_biohub_esmc_sae_profile_resume_reuses_accepted_rows(tmp_path: Path) -> None:
    write_foldcheck_report_fixture(tmp_path, accepted_candidate_ids={"wild_type", "thread_candidate_test"})

    materialize_biohub_esmc_sae_profile(
        repo_root=Path.cwd(),
        output_root=tmp_path,
        sequence_limit="1",
        sae_model=_FIXTURE_SAE_MODEL,
        biohub_client=FakeBiohubEsmcClient(),
        retrieved_at="2026-06-25T00:00:00Z",
    )
    fake_client = FakeBiohubEsmcClient()
    result = materialize_biohub_esmc_sae_profile(
        repo_root=Path.cwd(),
        output_root=tmp_path,
        sequence_limit="all",
        resume_existing=True,
        max_new_requests=1,
        sae_model=_FIXTURE_SAE_MODEL,
        biohub_client=fake_client,
        retrieved_at="2026-06-25T00:00:00Z",
    )

    profile_rows = pq.read_table(result.profile_path).to_pylist()
    statuses = {row["candidate_id"]: row["status"] for row in profile_rows}
    assert statuses == {"wild_type": "accepted", "thread_candidate_test": "accepted"}
    assert fake_client.requested_sequences == ["AAAE"]


def test_biohub_esmc_sae_profile_final_run_rejects_timeout_error_row(tmp_path: Path) -> None:
    write_foldcheck_report_fixture(
        tmp_path,
        accepted_candidate_ids={"wild_type", "thread_candidate_test"},
    )

    with pytest.raises(ValueError, match="requires every selected sequence to be accepted"):
        materialize_biohub_esmc_sae_profile(
            repo_root=Path.cwd(),
            output_root=tmp_path,
            sequence_limit="all",
            sae_model=_FIXTURE_SAE_MODEL,
            biohub_client=TimeoutOnceBiohubEsmcClient(),
            retrieved_at="2026-06-25T00:00:00Z",
        )


def test_biohub_esmc_sae_profile_raises_on_malformed_logits_schema(tmp_path: Path) -> None:
    write_foldcheck_report_fixture(tmp_path, accepted_candidate_ids={"wild_type"})

    with pytest.raises(ValueError, match="sae_outputs"):
        materialize_biohub_esmc_sae_profile(
            repo_root=Path.cwd(),
            output_root=tmp_path,
            sequence_limit="1",
            sae_model=_FIXTURE_SAE_MODEL,
            biohub_client=MalformedBiohubEsmcClient(),
            retrieved_at="2026-06-25T00:00:00Z",
        )
