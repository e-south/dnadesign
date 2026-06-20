"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/source_sequences/test_materialization.py

Source-sequence bundle materialization tests for Eco1 RT repack.
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dnadesign.aligner.msa import load_fasta_records
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences import (
    materialize_source_sequence_bundles,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.structure import (
    materialize_structure_authority,
)
from dnadesign.studies.units.eco1_rt_repack.tests._helpers import repo_root
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.source_sequences._fixtures import (
    TARGET_ROW_ID,
    mutate,
    target_sequence,
    write_source_cache,
)


def test_source_sequence_materializer_writes_profile_fastas_and_manifests(tmp_path: Path) -> None:
    materialize_structure_authority(repo_root=repo_root(), output_root=tmp_path)
    target = target_sequence(tmp_path)
    cache_root = write_source_cache(tmp_path, target)

    result = materialize_source_sequence_bundles(
        repo_root=repo_root(),
        output_root=tmp_path,
        source_cache_root=cache_root,
    )

    bundle_manifest = yaml.safe_load(result.bundle_manifest_path.read_text(encoding="utf-8"))
    assert bundle_manifest["schema_id"] == "eco1_rt_repack.conservation_source_sequence_bundle.index"
    assert bundle_manifest["profile_ids"] == ["broad_retron_rt", "eco1_like_retron_rt"]

    broad_records = load_fasta_records(result.fasta_paths["broad_retron_rt"])
    assert list(broad_records)[0] == TARGET_ROW_ID
    assert broad_records[TARGET_ROW_ID] == target
    assert broad_records["broad_ncbi_1"] == target
    assert broad_records["broad_bvbrc_1"] == mutate(target, 8, "A")

    broad_manifest = yaml.safe_load(result.manifest_paths["broad_retron_rt"].read_text(encoding="utf-8"))
    assert broad_manifest["schema_id"] == "eco1_rt_repack.conservation_source_sequence_bundle.profile"
    assert broad_manifest["status"] == "materialized"
    assert broad_manifest["target_row_id"] == TARGET_ROW_ID
    assert broad_manifest["included_record_count"] == 2
    assert broad_manifest["excluded_record_count"] == 1
    assert broad_manifest["excluded_records"][0]["exclusion_reason"] == "provider_unresolved"


def test_source_sequence_materializer_rejects_missing_provider_sequence(tmp_path: Path) -> None:
    materialize_structure_authority(repo_root=repo_root(), output_root=tmp_path)
    cache_root = write_source_cache(tmp_path, target_sequence(tmp_path), omit_accession="WP_BROAD_1")

    with pytest.raises(ValueError, match="missing provider FASTA record"):
        materialize_source_sequence_bundles(
            repo_root=repo_root(),
            output_root=tmp_path,
            source_cache_root=cache_root,
        )


def test_source_sequence_materializer_rejects_undeclared_provider(tmp_path: Path) -> None:
    materialize_structure_authority(repo_root=repo_root(), output_root=tmp_path)
    cache_root = write_source_cache(tmp_path, target_sequence(tmp_path), provider_override="uncontracted_provider")

    with pytest.raises(ValueError, match="not declared"):
        materialize_source_sequence_bundles(
            repo_root=repo_root(),
            output_root=tmp_path,
            source_cache_root=cache_root,
        )


def test_source_sequence_materializer_rejects_operator_supplied_target_row(tmp_path: Path) -> None:
    materialize_structure_authority(repo_root=repo_root(), output_root=tmp_path)
    cache_root = write_source_cache(tmp_path, target_sequence(tmp_path), record_id_override=TARGET_ROW_ID)

    with pytest.raises(ValueError, match="target row is inserted by the materializer"):
        materialize_source_sequence_bundles(
            repo_root=repo_root(),
            output_root=tmp_path,
            source_cache_root=cache_root,
        )


def test_source_sequence_materializer_rejects_exclusion_without_reason(tmp_path: Path) -> None:
    materialize_structure_authority(repo_root=repo_root(), output_root=tmp_path)
    cache_root = write_source_cache(tmp_path, target_sequence(tmp_path), omit_exclusion_reason=True)

    with pytest.raises(ValueError, match="exclusion_reason"):
        materialize_source_sequence_bundles(
            repo_root=repo_root(),
            output_root=tmp_path,
            source_cache_root=cache_root,
        )
