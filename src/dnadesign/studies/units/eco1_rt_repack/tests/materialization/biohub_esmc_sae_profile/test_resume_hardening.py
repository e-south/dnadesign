"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/biohub_esmc_sae_profile/test_resume_hardening.py

Eco1 Biohub ESMC SAE-profile resume-hardening tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.biohub_esmc_sae_profile import (
    materialize_biohub_esmc_sae_profile,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.atlas_semantic_profile._fixtures import (
    write_foldcheck_report_fixture,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.biohub_esmc_sae_profile.fixtures import (
    FakeBiohubEsmcClient,
)

_FIXTURE_SAE_MODEL = "fixture-sae-model"


def test_biohub_esmc_sae_profile_resume_requeries_partial_cached_rows(tmp_path: Path) -> None:
    write_foldcheck_report_fixture(tmp_path, accepted_candidate_ids={"wild_type"})

    materialize_biohub_esmc_sae_profile(
        repo_root=Path.cwd(),
        output_root=tmp_path,
        sequence_limit="1",
        sae_model=_FIXTURE_SAE_MODEL,
        biohub_client=FakeBiohubEsmcClient(),
        retrieved_at="2026-06-25T00:00:00Z",
    )
    residue_path = tmp_path / "biohub_esmc_residue_features.parquet"
    residue_table = pq.read_table(residue_path)
    pq.write_table(residue_table.slice(0, 0).replace_schema_metadata(residue_table.schema.metadata), residue_path)

    fake_client = FakeBiohubEsmcClient()
    result = materialize_biohub_esmc_sae_profile(
        repo_root=Path.cwd(),
        output_root=tmp_path,
        sequence_limit="1",
        resume_existing=True,
        max_new_requests=1,
        sae_model=_FIXTURE_SAE_MODEL,
        biohub_client=fake_client,
        retrieved_at="2026-06-25T00:00:00Z",
    )

    assert fake_client.requested_sequences == ["AAAA"]
    assert pq.read_table(result.residue_features_path).num_rows == 3
