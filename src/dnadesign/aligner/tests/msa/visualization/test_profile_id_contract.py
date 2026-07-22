"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/aligner/tests/msa/visualization/test_profile_id_contract.py

Tests for profile-id filename safety in MSA visualization requests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dnadesign.aligner.msa.visualization import (
    MsaVisualizationRequest,
    materialize_msa_visualizations,
)
from dnadesign.aligner.tests.msa.visualization._fixtures import (
    TARGET,
    target_hash,
    write_alignment_inputs,
)


@pytest.mark.parametrize(
    "profile_id",
    (
        "../escape",
        "nested/profile",
        r"nested\profile",
        "/absolute/profile",
        ".",
        "..",
        "profile..escape",
        " profile_a",
        "profile_a ",
    ),
)
def test_visualization_rejects_profile_ids_that_are_not_file_safe_stems(tmp_path: Path, profile_id: str) -> None:
    alignment_root = tmp_path / "alignments"
    alignment_root.mkdir()

    with pytest.raises(ValueError, match="profile_ids must be file-safe stems"):
        MsaVisualizationRequest(
            alignment_root=alignment_root,
            output_root=tmp_path / "visualizations",
            profile_ids=(profile_id,),
            target_row_id="target",
            target_sequence_hash=target_hash(TARGET),
        )


def test_visualization_accepts_current_eco1_profile_id_stems(tmp_path: Path) -> None:
    profile_ids = ("ec86_clade9_conservation_v1", "ec86_iia3_cluster42_1_conservation_v1")
    alignment_root = write_alignment_inputs(tmp_path, profile_ids=profile_ids)

    result = materialize_msa_visualizations(
        MsaVisualizationRequest(
            alignment_root=alignment_root,
            output_root=tmp_path / "visualizations",
            profile_ids=profile_ids,
            target_row_id="target",
            target_sequence_hash=target_hash(TARGET),
        )
    )

    assert result.profile_ids == profile_ids
    assert sorted(result.profile_qc_paths) == sorted(profile_ids)
