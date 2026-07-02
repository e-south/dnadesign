"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/biohub_esmc_sae_profile/test_feature_description_enrichment.py

Eco1 Biohub ESMC SAE feature-description enrichment tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.biohub_esmc_sae_profile import (
    enrich_existing_biohub_esmc_feature_catalog,
    materialize_biohub_esmc_sae_profile,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.atlas_semantic_profile._fixtures import (
    write_foldcheck_report_fixture,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.biohub_esmc_sae_profile.fixtures import (
    FakeBiohubEsmcClient,
    FakeFeatureDescriptionClient,
)
from dnadesign.thread.adapters.biohub_esmc import FEATURE_DESCRIPTION_SAE_MODEL

_FIXTURE_SAE_MODEL = "fixture-sae-model"


def test_biohub_esmc_sae_profile_can_enrich_exact_dictionary_feature_descriptions(tmp_path: Path) -> None:
    write_foldcheck_report_fixture(tmp_path, accepted_candidate_ids={"wild_type"})
    feature_description_client = FakeFeatureDescriptionClient()

    result = materialize_biohub_esmc_sae_profile(
        repo_root=Path.cwd(),
        output_root=tmp_path,
        sequence_limit="1",
        sae_model=FEATURE_DESCRIPTION_SAE_MODEL,
        biohub_client=FakeBiohubEsmcClient(),
        fetch_feature_descriptions=True,
        feature_description_limit=2,
        feature_description_client=feature_description_client,
        retrieved_at="2026-06-25T00:00:00Z",
    )

    descriptions = _descriptions(result.feature_catalog_path)
    assert descriptions == {
        0: "Fixture exact-dictionary description for F0.",
        1: "Fixture exact-dictionary description for F1.",
    }
    assert feature_description_client.requested == [0, 1]
    manifest_text = result.request_manifest_path.read_text(encoding="utf-8")
    assert "status: enriched" in manifest_text
    assert "enriched_feature_count: 2" in manifest_text


def test_biohub_esmc_sae_profile_can_enrich_existing_catalog_without_rebuilding_tables(tmp_path: Path) -> None:
    write_foldcheck_report_fixture(tmp_path, accepted_candidate_ids={"wild_type"})
    materialized = materialize_biohub_esmc_sae_profile(
        repo_root=Path.cwd(),
        output_root=tmp_path,
        sequence_limit="1",
        sae_model=FEATURE_DESCRIPTION_SAE_MODEL,
        biohub_client=FakeBiohubEsmcClient(),
        retrieved_at="2026-06-25T00:00:00Z",
    )
    residue_mtime = materialized.residue_features_path.stat().st_mtime_ns
    feature_description_client = FakeFeatureDescriptionClient()

    result = enrich_existing_biohub_esmc_feature_catalog(
        repo_root=Path.cwd(),
        output_root=tmp_path,
        sae_model=FEATURE_DESCRIPTION_SAE_MODEL,
        feature_description_limit=2,
        feature_description_client=feature_description_client,
        retrieved_at="2026-06-25T00:01:00Z",
    )

    assert _descriptions(result.feature_catalog_path) == {
        0: "Fixture exact-dictionary description for F0.",
        1: "Fixture exact-dictionary description for F1.",
    }
    assert feature_description_client.requested == [0, 1]
    assert materialized.residue_features_path.stat().st_mtime_ns == residue_mtime
    manifest_text = result.manifest_path.read_text(encoding="utf-8")
    assert "schema_id: eco1_rt.biohub_esmc.feature_description_enrichment" in manifest_text
    assert "enriched_feature_count: 2" in manifest_text


def test_biohub_esmc_sae_profile_rejects_incompatible_feature_description_fetch(tmp_path: Path) -> None:
    write_foldcheck_report_fixture(tmp_path, accepted_candidate_ids={"wild_type"})

    with pytest.raises(ValueError, match="source-backed only"):
        materialize_biohub_esmc_sae_profile(
            repo_root=Path.cwd(),
            output_root=tmp_path,
            sequence_limit="1",
            sae_model=_FIXTURE_SAE_MODEL,
            biohub_client=FakeBiohubEsmcClient(),
            fetch_feature_descriptions=True,
            feature_description_client=FakeFeatureDescriptionClient(),
            retrieved_at="2026-06-25T00:00:00Z",
        )


def _descriptions(path: Path) -> dict[int, str]:
    rows = pq.read_table(path).to_pylist()
    return {row["feature_index"]: row["description"] for row in rows if row["description"]}
