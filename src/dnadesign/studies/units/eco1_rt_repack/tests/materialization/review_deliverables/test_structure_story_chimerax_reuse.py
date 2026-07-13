"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/test_structure_story_chimerax_reuse.py

ChimeraX structure-story render reuse tests for Eco1 RT review deliverables.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.communication_visuals import (  # noqa: E501
    candidate_cycle,
    chimerax_story,
)


def test_current_structure_story_render_is_reused_when_render_is_requested(tmp_path: Path, monkeypatch) -> None:
    script_path = tmp_path / "story.cxc"
    reference_path = tmp_path / "reference.pdb"
    movie_path = tmp_path / "story.mp4"
    render_manifest_path = tmp_path / "render.yaml"
    script_path.write_text("exit\n", encoding="utf-8")
    reference_path.write_text("END\n", encoding="utf-8")
    movie_path.write_bytes(b"mp4")
    render_manifest_path.write_text(
        yaml.safe_dump(
            {
                "script_hash": "sha256:" + chimerax_story.sha256(script_path),
                "reference_hash": "sha256:" + chimerax_story.sha256(reference_path),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        chimerax_story,
        "run_chimerax_script",
        lambda **_kwargs: pytest.fail("current render should not launch ChimeraX"),
    )

    status, reason = chimerax_story.materialize_chimerax_outputs(
        script_path=script_path,
        reference_structure_path=reference_path,
        movie_path=movie_path,
        frame_directory=tmp_path / "frames",
        render_manifest_path=render_manifest_path,
        render_chimerax=True,
    )

    assert (status, reason) == ("rendered", "")


def test_current_candidate_cycle_is_reused_when_render_is_requested(tmp_path: Path, monkeypatch) -> None:
    script_path = tmp_path / "candidate.cxc"
    source_path = tmp_path / "selection.parquet"
    movie_path = tmp_path / "candidate.mp4"
    render_manifest_path = tmp_path / "candidate-render.yaml"
    script_path.write_text("exit\n", encoding="utf-8")
    source_path.write_bytes(b"selection")
    movie_path.write_bytes(b"mp4")
    input_hashes = candidate_cycle.file_hashes(
        {"candidate_selection_panel": source_path, "candidate_cycle_script": script_path}
    )
    render_manifest_path.write_text(yaml.safe_dump({"input_hashes": input_hashes}), encoding="utf-8")
    monkeypatch.setattr(
        candidate_cycle,
        "run_chimerax_script",
        lambda **_kwargs: pytest.fail("current render should not launch ChimeraX"),
    )

    status, reason = candidate_cycle.materialize_candidate_cycle_movie(
        script_path=script_path,
        movie_path=movie_path,
        frame_directory=tmp_path / "frames",
        render_manifest_path=render_manifest_path,
        source_paths={"candidate_selection_panel": source_path},
        render_chimerax=True,
    )

    assert (status, reason) == ("rendered", "")
