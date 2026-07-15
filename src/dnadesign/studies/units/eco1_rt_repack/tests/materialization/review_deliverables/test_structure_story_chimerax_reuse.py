"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/test_structure_story_chimerax_reuse.py

Shared ChimeraX movie reuse and stale-output tests for Eco1 review deliverables.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.communication_visuals import (  # noqa: E501
    movie_runtime,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.manifest import (
    file_hashes,
    sha256,
)


def _spec() -> movie_runtime.MovieRenderSpec:
    return movie_runtime.MovieRenderSpec(
        schema_id="eco1_rt.test_movie_render",
        schema_version=1,
        renderer="ChimeraX test PNG saves",
        output_key="test_movie",
        frame_width=1280,
        frame_height=720,
        frame_rate=24,
        frames_per_scene=1,
        hold_frames_per_scene=1,
    )


def test_current_movie_render_is_reused_without_launching_chimerax(tmp_path: Path, monkeypatch) -> None:
    script_path = tmp_path / "story.cxc"
    source_path = tmp_path / "reference.pdb"
    movie_path = tmp_path / "story.mp4"
    render_manifest_path = tmp_path / "render.yaml"
    script_path.write_text("exit\n", encoding="utf-8")
    source_path.write_text("END\n", encoding="utf-8")
    movie_path.write_bytes(b"mp4")
    spec = _spec()
    input_hashes = file_hashes({"reference": source_path, "movie_script": script_path})
    render_manifest_path.write_text(
        yaml.safe_dump(
            {
                "schema_id": spec.schema_id,
                "schema_version": spec.schema_version,
                "status": "rendered",
                "input_hashes": input_hashes,
                "movie_encoding": {**spec.encoding_contract(), "raw_frame_count": 1, "frame_count": 2},
                "output": {
                    "key": spec.output_key,
                    "path": movie_path.name,
                    "sha256": "sha256:" + sha256(movie_path),
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        movie_runtime,
        "run_chimerax_script",
        lambda **_kwargs: pytest.fail("current render should not launch ChimeraX"),
    )

    status, reason = movie_runtime.materialize_chimerax_movie(
        script_path=script_path,
        movie_path=movie_path,
        frame_directory=tmp_path / "frames",
        render_manifest_path=render_manifest_path,
        log_path=tmp_path / "render.log",
        source_paths={"reference": source_path},
        render_requested=True,
        spec=spec,
        expected_raw_frame_count=1,
    )

    assert (status, reason) == ("rendered", "")


@pytest.mark.parametrize("tampered_field", ["status", "output_key", "output_path", "output_digest"])
def test_movie_reuse_rejects_inexact_render_manifest(tmp_path: Path, tampered_field: str) -> None:
    script_path = tmp_path / "story.cxc"
    source_path = tmp_path / "reference.pdb"
    movie_path = tmp_path / "story.mp4"
    render_manifest_path = tmp_path / "render.yaml"
    script_path.write_text("exit\n", encoding="utf-8")
    source_path.write_text("END\n", encoding="utf-8")
    movie_path.write_bytes(b"mp4")
    spec = _spec()
    payload = {
        "schema_id": spec.schema_id,
        "schema_version": spec.schema_version,
        "status": "rendered",
        "input_hashes": file_hashes({"reference": source_path, "movie_script": script_path}),
        "movie_encoding": {**spec.encoding_contract(), "raw_frame_count": 1, "frame_count": 2},
        "output": {
            "key": spec.output_key,
            "path": movie_path.name,
            "sha256": "sha256:" + sha256(movie_path),
        },
    }
    if tampered_field == "status":
        payload["status"] = "errored"
    elif tampered_field == "output_key":
        payload["output"]["key"] = "another_movie"
    elif tampered_field == "output_path":
        payload["output"]["path"] = "another.mp4"
    else:
        payload["output"]["sha256"] = "sha256:" + "0" * 64
    render_manifest_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    status, _reason = movie_runtime.materialize_chimerax_movie(
        script_path=script_path,
        movie_path=movie_path,
        frame_directory=tmp_path / "frames",
        render_manifest_path=render_manifest_path,
        log_path=tmp_path / "render.log",
        source_paths={"reference": source_path},
        render_requested=False,
        spec=spec,
        expected_raw_frame_count=1,
    )

    assert status == "skipped_stale_optional_render_retained"
    assert movie_path.exists()
    assert render_manifest_path.exists()


def test_stale_movie_is_retained_but_staging_frames_are_removed_when_target_is_not_requested(tmp_path: Path) -> None:
    script_path = tmp_path / "story.cxc"
    source_path = tmp_path / "reference.pdb"
    movie_path = tmp_path / "story.mp4"
    render_manifest_path = tmp_path / "render.yaml"
    log_path = tmp_path / "render.log"
    frame_directory = tmp_path / "frames"
    script_path.write_text("exit\n", encoding="utf-8")
    source_path.write_text("END\n", encoding="utf-8")
    movie_path.write_bytes(b"stale")
    render_manifest_path.write_text("schema_id: stale\n", encoding="utf-8")
    log_path.write_text("stale log\n", encoding="utf-8")
    frame_directory.mkdir()
    (frame_directory / "frame-00001.png").write_bytes(b"stale frame")

    status, reason = movie_runtime.materialize_chimerax_movie(
        script_path=script_path,
        movie_path=movie_path,
        frame_directory=frame_directory,
        render_manifest_path=render_manifest_path,
        log_path=log_path,
        source_paths={"reference": source_path},
        render_requested=False,
        spec=_spec(),
        expected_raw_frame_count=1,
    )

    assert status == "skipped_stale_optional_render_retained"
    assert "stale" in reason.lower()
    assert movie_path.exists()
    assert render_manifest_path.exists()
    assert log_path.exists()
    assert not frame_directory.exists()
