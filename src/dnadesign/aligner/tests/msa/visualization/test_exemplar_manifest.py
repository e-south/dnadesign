"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/aligner/tests/msa/visualization/test_exemplar_manifest.py

Regression tests for exemplar-window manifest freshness.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

from dnadesign.aligner.msa.visualization import (
    MsaVisualizationRequest,
    materialize_msa_visualizations,
)
from dnadesign.aligner.tests.msa.visualization._fixtures import (
    TARGET,
    target_hash,
    write_alignment_inputs,
    write_annotation_tracks,
    write_exemplar_rows,
)


def test_visualization_ignores_stale_exemplar_svgs_from_prior_run(tmp_path: Path) -> None:
    alignment_root = write_alignment_inputs(tmp_path, profile_ids=("profile_a", "profile_b"))
    output_root = tmp_path / "visualizations"
    annotation_tracks = write_annotation_tracks(tmp_path)

    materialize_msa_visualizations(
        MsaVisualizationRequest(
            alignment_root=alignment_root,
            output_root=output_root,
            profile_ids=("profile_a", "profile_b"),
            target_row_id="target",
            target_sequence_hash=target_hash(TARGET),
            annotation_tracks_yaml=annotation_tracks,
            exemplar_rows_yaml=write_exemplar_rows(tmp_path, profile_ids=("profile_a", "profile_b")),
        )
    )

    result = materialize_msa_visualizations(
        MsaVisualizationRequest(
            alignment_root=alignment_root,
            output_root=output_root,
            profile_ids=("profile_a", "profile_b"),
            target_row_id="target",
            target_sequence_hash=target_hash(TARGET),
            annotation_tracks_yaml=annotation_tracks,
            exemplar_rows_yaml=write_exemplar_rows(tmp_path, profile_ids=("profile_a",)),
        )
    )

    html_report = result.html_report_path.read_text(encoding="utf-8")
    assert "profile_a.exemplar_windows.svg" in html_report
    assert "profile_b.exemplar_windows.svg" not in html_report
    index = yaml.safe_load(result.index_manifest_path.read_text(encoding="utf-8"))
    assert set(index["profile_exemplar_svg_paths"]) == {"profile_a"}
    assert set(result.profile_exemplar_svg_paths) == {"profile_a"}
