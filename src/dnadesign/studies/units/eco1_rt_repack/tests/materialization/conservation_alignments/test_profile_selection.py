"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/conservation_alignments/test_profile_selection.py

Selected-profile behavior for Eco1 conservation-alignment materialization.

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

from ._runners import recording_copy_runner

pytestmark = pytest.mark.skipif(
    not ec86kit_source_artifacts_available(), reason="requires sibling ec86kit structure-authority artifacts"
)


def test_alignment_materializer_scopes_sufficiency_to_selected_profile(tmp_path: Path) -> None:
    materialize_structure_authority(repo_root=repo_root(), output_root=tmp_path)
    cache_root = write_sufficient_source_cache(tmp_path, target_sequence(tmp_path))
    source_result = materialize_source_sequence_bundles(
        repo_root=repo_root(),
        output_root=tmp_path,
        source_cache_root=cache_root,
    )
    non_selected_manifest_path = source_result.manifest_paths["ec86_clade9_conservation_v1"]
    non_selected_manifest = yaml.safe_load(non_selected_manifest_path.read_text(encoding="utf-8"))
    non_selected_manifest["included_records"] = []
    non_selected_manifest["included_record_count"] = 0
    non_selected_manifest_path.write_text(
        yaml.safe_dump(non_selected_manifest, sort_keys=False),
        encoding="utf-8",
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
