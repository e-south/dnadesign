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
from typing import Any

import pyarrow.parquet as pq
import torch

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.biohub_esmc_sae_profile import (
    materialize_biohub_esmc_sae_profile,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.atlas_semantic_profile._fixtures import (
    write_foldcheck_report_fixture,
)
from dnadesign.thread.adapters.biohub_esmc import DEFAULT_ESMC_SAE_MODEL, BiohubCredential


def test_biohub_esmc_sae_profile_materializes_wt_only_smoke(tmp_path: Path) -> None:
    write_foldcheck_report_fixture(tmp_path, accepted_candidate_ids={"wild_type"})

    result = materialize_biohub_esmc_sae_profile(
        repo_root=Path.cwd(),
        output_root=tmp_path,
        sequence_limit="1",
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


def test_biohub_esmc_sae_profile_caps_new_requests_with_explicit_rows(tmp_path: Path) -> None:
    write_foldcheck_report_fixture(tmp_path, accepted_candidate_ids={"wild_type", "thread_candidate_test"})

    result = materialize_biohub_esmc_sae_profile(
        repo_root=Path.cwd(),
        output_root=tmp_path,
        sequence_limit="all",
        max_new_requests=1,
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
        biohub_client=fake_client,
        retrieved_at="2026-06-25T00:00:00Z",
    )

    profile_rows = pq.read_table(result.profile_path).to_pylist()
    statuses = {row["candidate_id"]: row["status"] for row in profile_rows}
    assert statuses == {"wild_type": "accepted", "thread_candidate_test": "accepted"}
    assert fake_client.requested_sequences == ["AAAE"]


def test_biohub_esmc_sae_profile_writes_timeout_error_row(tmp_path: Path) -> None:
    write_foldcheck_report_fixture(
        tmp_path,
        accepted_candidate_ids={"wild_type", "thread_candidate_test"},
    )

    result = materialize_biohub_esmc_sae_profile(
        repo_root=Path.cwd(),
        output_root=tmp_path,
        sequence_limit="all",
        biohub_client=TimeoutOnceBiohubEsmcClient(),
        retrieved_at="2026-06-25T00:00:00Z",
    )

    profile_rows = pq.read_table(result.profile_path).to_pylist()
    statuses = {row["candidate_id"]: row["status"] for row in profile_rows}
    failure_reasons = {row["candidate_id"]: row["failure_reason"] for row in profile_rows}
    assert statuses == {
        "wild_type": "accepted",
        "thread_candidate_test": "errored",
    }
    assert "read operation timed out" in failure_reasons["thread_candidate_test"]


class FakeBiohubEsmcClient:
    def __init__(self) -> None:
        self.credential = BiohubCredential(key_label="bu-dunlop-lab", token="fixture-secret")
        self.requested_sequences: list[str] = []

    def logits_for_sequence(
        self,
        sequence: str,
        *,
        model: str,
        sae_model: str,
        normalize_features: bool,
    ) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
        del model, normalize_features
        normalized = sequence.strip().upper()
        self.requested_sequences.append(normalized[:4])
        tokens = [0, *range(1, len(normalized) + 1), 2]
        tensor = torch.zeros((len(normalized) + 2, 16), dtype=torch.float32)
        tensor[1, 3] = 1.5
        tensor[2, 7] = 2.0
        tensor[len(normalized), 7] = 4.0
        return (
            {"outputs": {"sequence": tokens}, "potential_sequence_of_concern": False},
            {"sae_outputs": {sae_model or DEFAULT_ESMC_SAE_MODEL: tensor}, "logits": None, "embeddings": None},
            tokens,
        )


class TimeoutOnceBiohubEsmcClient(FakeBiohubEsmcClient):
    def logits_for_sequence(
        self,
        sequence: str,
        *,
        model: str,
        sae_model: str,
        normalize_features: bool,
    ) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
        if len(self.requested_sequences) == 1:
            self.requested_sequences.append(sequence.strip().upper()[:4])
            raise TimeoutError("The read operation timed out")
        return super().logits_for_sequence(
            sequence,
            model=model,
            sae_model=sae_model,
            normalize_features=normalize_features,
        )
