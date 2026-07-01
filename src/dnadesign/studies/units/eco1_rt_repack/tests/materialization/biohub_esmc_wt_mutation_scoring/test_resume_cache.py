"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/biohub_esmc_wt_mutation_scoring/test_resume_cache.py

Eco1 Biohub ESMC mutation-scoring resume-cache tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

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


def test_wt_mutation_scoring_resume_rejects_stale_position_schema(tmp_path: Path) -> None:
    write_foldcheck_report_fixture(tmp_path, accepted_candidate_ids={"wild_type"})
    write_mask_set(tmp_path / "mask_set.yaml", length=4)

    first = materialize_biohub_esmc_wt_mutation_scoring(
        repo_root=Path.cwd(),
        output_root=tmp_path,
        positions="1",
        max_new_requests=1,
        biohub_client=FakeSequenceLogitsClient(),
        retrieved_at="2026-06-27T00:00:00Z",
    )
    rewrite_position_table_with_old_fraction_name(first.position_entropy_path)

    with pytest.raises(ValueError, match="stale mutation-scoring cache"):
        materialize_biohub_esmc_wt_mutation_scoring(
            repo_root=Path.cwd(),
            output_root=tmp_path,
            positions="1",
            resume_existing=True,
            max_new_requests=0,
            biohub_client=FakeSequenceLogitsClient(),
            retrieved_at="2026-06-27T00:00:00Z",
        )


def test_wt_mutation_scoring_resume_rejects_null_accepted_metric(tmp_path: Path) -> None:
    write_foldcheck_report_fixture(tmp_path, accepted_candidate_ids={"wild_type"})
    write_mask_set(tmp_path / "mask_set.yaml", length=4)

    first = materialize_biohub_esmc_wt_mutation_scoring(
        repo_root=Path.cwd(),
        output_root=tmp_path,
        positions="1",
        max_new_requests=1,
        biohub_client=FakeSequenceLogitsClient(),
        retrieved_at="2026-06-27T00:00:00Z",
    )
    rewrite_position_table_with_null_alternate_fraction(first.position_entropy_path)

    with pytest.raises(ValueError, match="stale mutation-scoring cache"):
        materialize_biohub_esmc_wt_mutation_scoring(
            repo_root=Path.cwd(),
            output_root=tmp_path,
            positions="1",
            resume_existing=True,
            max_new_requests=0,
            biohub_client=FakeSequenceLogitsClient(),
            retrieved_at="2026-06-27T00:00:00Z",
        )
