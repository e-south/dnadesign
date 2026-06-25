"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/conservation_alignments/test_materialization.py

Conservation-alignment bundle materialization tests for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dnadesign.aligner.msa import MsaRequest
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.conservation_alignments import (
    materialize_conservation_alignment_bundles,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences import (
    materialize_source_sequence_bundles,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.structure import (
    materialize_structure_authority,
)
from dnadesign.studies.units.eco1_rt_repack.tests._helpers import ec86kit_source_artifacts_available, repo_root
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.source_sequences._fixtures import (
    target_sequence,
    write_sufficient_source_cache,
)

from ._runners import forbidden_runner, recording_copy_runner

pytestmark = pytest.mark.skipif(
    not ec86kit_source_artifacts_available(), reason="requires sibling ec86kit structure-authority artifacts"
)


def test_alignment_materializer_fails_before_msa_when_source_sufficiency_fails(tmp_path: Path) -> None:
    materialize_structure_authority(repo_root=repo_root(), output_root=tmp_path)

    with pytest.raises(ValueError, match="source sequence sufficiency failed"):
        materialize_conservation_alignment_bundles(
            repo_root=repo_root(),
            output_root=tmp_path,
            source_cache_root=tmp_path / "missing_cache",
            source_bundle_root=tmp_path / "missing_sources",
            msa_runner=forbidden_runner,
        )


def test_alignment_materializer_runs_each_profile_and_writes_index_manifest(tmp_path: Path) -> None:
    materialize_structure_authority(repo_root=repo_root(), output_root=tmp_path)
    cache_root = write_sufficient_source_cache(tmp_path, target_sequence(tmp_path))
    source_result = materialize_source_sequence_bundles(
        repo_root=repo_root(),
        output_root=tmp_path,
        source_cache_root=cache_root,
    )
    observed_requests: list[MsaRequest] = []

    result = materialize_conservation_alignment_bundles(
        repo_root=repo_root(),
        output_root=tmp_path,
        source_cache_root=cache_root,
        source_bundle_root=source_result.bundle_manifest_path.parent,
        msa_runner=recording_copy_runner(observed_requests),
    )

    assert sorted(result.aligned_fasta_paths) == [
        "ec86_clade9_conservation_v1",
        "ec86_iia3_cluster42_1_conservation_v1",
    ]
    assert len(observed_requests) == 2
    assert {request.target_row_id for request in observed_requests} == {"eco1_rt_ec86kit_reference"}
    assert {request.backend.backend_id for request in observed_requests} == {"clustalo"}
    assert {request.command_args for request in observed_requests} == {("--force", "--outfmt=fasta", "--threads=1")}
    assert {request.input_fasta.name for request in observed_requests} == {
        "ec86_clade9_conservation_v1.source.fasta",
        "ec86_iia3_cluster42_1_conservation_v1.source.fasta",
    }
    assert {request.output_fasta.name for request in observed_requests} == {
        "ec86_clade9_conservation_v1.aligned.fasta",
        "ec86_iia3_cluster42_1_conservation_v1.aligned.fasta",
    }

    bundle_manifest = yaml.safe_load(result.bundle_manifest_path.read_text(encoding="utf-8"))
    assert bundle_manifest["schema_id"] == "eco1_rt_repack.conservation_alignment_bundle.index"
    assert bundle_manifest["status"] == "materialized"
    assert bundle_manifest["profile_ids"] == ["ec86_clade9_conservation_v1", "ec86_iia3_cluster42_1_conservation_v1"]
    assert bundle_manifest["target_row_id"] == "eco1_rt_ec86kit_reference"
    assert bundle_manifest["alignment_manifests"]["ec86_clade9_conservation_v1"].endswith(
        "ec86_clade9_conservation_v1.aligned.manifest.yaml"
    )
    assert bundle_manifest["upstream_hashes"]["conservation_sources_yaml"].startswith("sha256:")


def test_alignment_materializer_can_run_one_declared_profile(tmp_path: Path) -> None:
    materialize_structure_authority(repo_root=repo_root(), output_root=tmp_path)
    cache_root = write_sufficient_source_cache(tmp_path, target_sequence(tmp_path))
    source_result = materialize_source_sequence_bundles(
        repo_root=repo_root(),
        output_root=tmp_path,
        source_cache_root=cache_root,
    )
    observed_requests: list[MsaRequest] = []

    result = materialize_conservation_alignment_bundles(
        repo_root=repo_root(),
        output_root=tmp_path,
        source_cache_root=cache_root,
        source_bundle_root=source_result.bundle_manifest_path.parent,
        profile_ids=("ec86_iia3_cluster42_1_conservation_v1",),
        msa_runner=recording_copy_runner(observed_requests),
    )

    assert list(result.aligned_fasta_paths) == ["ec86_iia3_cluster42_1_conservation_v1"]
    assert [request.run_label for request in observed_requests] == ["ec86_iia3_cluster42_1_conservation_v1"]
    assert [request.input_fasta.name for request in observed_requests] == [
        "ec86_iia3_cluster42_1_conservation_v1.source.fasta"
    ]
    bundle_manifest = yaml.safe_load(result.bundle_manifest_path.read_text(encoding="utf-8"))
    assert bundle_manifest["profile_ids"] == ["ec86_iia3_cluster42_1_conservation_v1"]
    assert set(bundle_manifest["profile_runs"][0]) >= {"profile_id", "elapsed_seconds"}


def test_alignment_materializer_rejects_unknown_profile_selection(tmp_path: Path) -> None:
    materialize_structure_authority(repo_root=repo_root(), output_root=tmp_path)
    cache_root = write_sufficient_source_cache(tmp_path, target_sequence(tmp_path))
    source_result = materialize_source_sequence_bundles(
        repo_root=repo_root(),
        output_root=tmp_path,
        source_cache_root=cache_root,
    )

    with pytest.raises(ValueError, match="Unknown conservation alignment profile"):
        materialize_conservation_alignment_bundles(
            repo_root=repo_root(),
            output_root=tmp_path,
            source_cache_root=cache_root,
            source_bundle_root=source_result.bundle_manifest_path.parent,
            profile_ids=("not_a_profile",),
            msa_runner=forbidden_runner,
        )
