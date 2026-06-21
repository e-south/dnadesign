"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/source_sequences/sufficiency/test_sufficiency.py

Source-sequence bundle sufficiency tests for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences import (
    materialize_source_sequence_bundles,
    validate_source_sequence_bundle_sufficiency,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.structure import (
    materialize_structure_authority,
)
from dnadesign.studies.units.eco1_rt_repack.tests._helpers import repo_root
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.source_sequences._fixtures import (
    target_sequence,
    write_source_cache,
    write_sufficient_source_cache,
)


def test_sufficiency_gate_accepts_realistic_source_bundle(tmp_path: Path) -> None:
    materialize_structure_authority(repo_root=repo_root(), output_root=tmp_path)
    cache_root = write_sufficient_source_cache(tmp_path, target_sequence(tmp_path))
    materialize_source_sequence_bundles(
        repo_root=repo_root(),
        output_root=tmp_path,
        source_cache_root=cache_root,
    )

    report = validate_source_sequence_bundle_sufficiency(
        repo_root=repo_root(),
        output_root=tmp_path,
        source_cache_root=cache_root,
    )

    assert report.passed is True
    assert report.issues == ()


def test_sufficiency_gate_rejects_undersized_fixture_like_bundle(tmp_path: Path) -> None:
    materialize_structure_authority(repo_root=repo_root(), output_root=tmp_path)
    cache_root = write_source_cache(tmp_path, target_sequence(tmp_path))
    materialize_source_sequence_bundles(
        repo_root=repo_root(),
        output_root=tmp_path,
        source_cache_root=cache_root,
    )

    report = validate_source_sequence_bundle_sufficiency(
        repo_root=repo_root(),
        output_root=tmp_path,
        source_cache_root=cache_root,
    )

    assert "eco1_rt.source_sequences.insufficient_included_records" in _check_ids(report)


def test_sufficiency_gate_rejects_placeholder_accessions(tmp_path: Path) -> None:
    materialize_structure_authority(repo_root=repo_root(), output_root=tmp_path)
    cache_root = write_sufficient_source_cache(
        tmp_path,
        target_sequence(tmp_path),
        placeholder_accession=True,
    )
    result = materialize_source_sequence_bundles(
        repo_root=repo_root(),
        output_root=tmp_path,
        source_cache_root=cache_root,
    )

    report = validate_source_sequence_bundle_sufficiency(
        repo_root=repo_root(),
        output_root=tmp_path,
        source_cache_root=cache_root,
        bundle_root=result.bundle_manifest_path.parent,
    )

    assert "eco1_rt.source_sequences.invalid_provider_accession" in _check_ids(report)


def test_sufficiency_gate_rejects_missing_source_cache_root(tmp_path: Path) -> None:
    materialize_structure_authority(repo_root=repo_root(), output_root=tmp_path)
    cache_root = write_sufficient_source_cache(tmp_path, target_sequence(tmp_path))
    result = materialize_source_sequence_bundles(
        repo_root=repo_root(),
        output_root=tmp_path,
        source_cache_root=cache_root,
    )

    report = validate_source_sequence_bundle_sufficiency(
        repo_root=repo_root(),
        output_root=tmp_path,
        source_cache_root=tmp_path / "missing_cache",
        bundle_root=result.bundle_manifest_path.parent,
    )

    assert "eco1_rt.source_sequences.source_cache_root_missing" in _check_ids(report)


def test_sufficiency_gate_rejects_mutated_provider_cache_hash(tmp_path: Path) -> None:
    materialize_structure_authority(repo_root=repo_root(), output_root=tmp_path)
    cache_root = write_sufficient_source_cache(tmp_path, target_sequence(tmp_path))
    result = materialize_source_sequence_bundles(
        repo_root=repo_root(),
        output_root=tmp_path,
        source_cache_root=cache_root,
    )
    provider_cache = cache_root / "provider_caches/ncbi_protein_efetch.fasta"
    provider_cache.write_text(provider_cache.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    report = validate_source_sequence_bundle_sufficiency(
        repo_root=repo_root(),
        output_root=tmp_path,
        source_cache_root=cache_root,
        bundle_root=result.bundle_manifest_path.parent,
    )

    assert "eco1_rt.source_sequences.provider_cache_hash_mismatch" in _check_ids(report)


def test_sufficiency_gate_rejects_mutated_source_records_hash(tmp_path: Path) -> None:
    materialize_structure_authority(repo_root=repo_root(), output_root=tmp_path)
    cache_root = write_sufficient_source_cache(tmp_path, target_sequence(tmp_path))
    result = materialize_source_sequence_bundles(
        repo_root=repo_root(),
        output_root=tmp_path,
        source_cache_root=cache_root,
    )
    source_records = cache_root / "source_records.yaml"
    source_records.write_text(source_records.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    report = validate_source_sequence_bundle_sufficiency(
        repo_root=repo_root(),
        output_root=tmp_path,
        source_cache_root=cache_root,
        bundle_root=result.bundle_manifest_path.parent,
    )

    assert "eco1_rt.source_sequences.source_records_hash_mismatch" in _check_ids(report)


def test_sufficiency_gate_rejects_profile_manifest_without_sequence_qc(tmp_path: Path) -> None:
    materialize_structure_authority(repo_root=repo_root(), output_root=tmp_path)
    cache_root = write_sufficient_source_cache(tmp_path, target_sequence(tmp_path))
    result = materialize_source_sequence_bundles(
        repo_root=repo_root(),
        output_root=tmp_path,
        source_cache_root=cache_root,
    )
    manifest_path = result.manifest_paths["ec86_clade9_conservation_v1"]
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["included_records"][0].pop("sequence_qc")
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    report = validate_source_sequence_bundle_sufficiency(
        repo_root=repo_root(),
        output_root=tmp_path,
        source_cache_root=cache_root,
        bundle_root=result.bundle_manifest_path.parent,
    )

    assert "eco1_rt.source_sequences.sequence_qc_missing" in _check_ids(report)


def _check_ids(report: object) -> set[str]:
    return {issue.check_id for issue in report.issues}
