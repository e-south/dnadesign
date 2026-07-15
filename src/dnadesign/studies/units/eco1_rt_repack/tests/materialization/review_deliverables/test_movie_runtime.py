"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/test_movie_runtime.py

Shared ChimeraX execution and frame-encoding tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from PIL import Image

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.communication_visuals import (  # noqa: E501
    movie_runtime,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.communication_visuals.pose import (  # noqa: E501
    CHIMERAX_HOLD_FRAMES_PER_SCENE,
    CHIMERAX_ROTATION_FRAMES_PER_SCENE,
)


def test_chimerax_path_quoting_is_absolute_and_escapes_quotes(tmp_path: Path) -> None:
    path = tmp_path / "structures" / 'candidate"1.pdb'

    assert movie_runtime.quoted_absolute_path(path) == f'"{path.resolve().as_posix().replace(chr(34), r"\"")}"'


def test_chimerax_communication_render_uses_graphical_cli_and_records_outputs(tmp_path: Path, monkeypatch) -> None:
    script_path = tmp_path / "story.cxc"
    reference_path = tmp_path / "reference.pdb"
    movie_path = tmp_path / "story.mp4"
    frame_directory = tmp_path / "frames"
    render_manifest_path = tmp_path / "render.yaml"
    script_path.write_text("exit\n", encoding="utf-8")
    reference_path.write_text("END\n", encoding="utf-8")
    observed: dict[str, object] = {}
    spec = movie_runtime.MovieRenderSpec(
        schema_id="eco1_rt.test_movie",
        schema_version=1,
        renderer="ChimeraX test renderer",
        output_key="test_movie",
        frame_width=1280,
        frame_height=720,
        frame_rate=24,
        frames_per_scene=CHIMERAX_ROTATION_FRAMES_PER_SCENE,
        hold_frames_per_scene=CHIMERAX_HOLD_FRAMES_PER_SCENE,
    )

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout="rendered", stderr="")

    def fake_encode_movie_frames(**kwargs):
        assert kwargs["frame_directory"] == frame_directory
        assert kwargs["frames_per_scene"] == CHIMERAX_ROTATION_FRAMES_PER_SCENE
        assert kwargs["frame_width"] == 1280
        assert kwargs["frame_height"] == 720
        kwargs["movie_path"].write_bytes(b"mp4")
        return 9 * (CHIMERAX_ROTATION_FRAMES_PER_SCENE + CHIMERAX_HOLD_FRAMES_PER_SCENE)

    monkeypatch.setattr(movie_runtime, "find_chimerax", lambda: "/Applications/ChimeraX.app/ChimeraX")
    monkeypatch.setattr(movie_runtime.subprocess, "run", fake_run)
    monkeypatch.setattr(movie_runtime, "encode_movie_frames", fake_encode_movie_frames)

    status, reason = movie_runtime.materialize_chimerax_movie(
        script_path=script_path,
        movie_path=movie_path,
        frame_directory=frame_directory,
        render_manifest_path=render_manifest_path,
        log_path=tmp_path / "render.log",
        source_paths={"reference_structure": reference_path},
        render_requested=True,
        spec=spec,
        expected_raw_frame_count=9 * CHIMERAX_ROTATION_FRAMES_PER_SCENE,
    )

    assert status == "rendered"
    assert reason == ""
    assert observed["command"] == [
        "/Applications/ChimeraX.app/ChimeraX",
        "--exit",
        "--script",
        str(script_path),
    ]
    render_payload = yaml.safe_load(render_manifest_path.read_text(encoding="utf-8"))
    assert render_payload["movie_encoding"]["frame_count"] == 9 * (
        CHIMERAX_ROTATION_FRAMES_PER_SCENE + CHIMERAX_HOLD_FRAMES_PER_SCENE
    )
    assert render_payload["movie_encoding"]["renderer"] == "ChimeraX test renderer"
    assert render_payload["movie_encoding"]["width"] == 1280
    assert render_payload["movie_encoding"]["height"] == 720
    assert render_payload["movie_encoding"]["raw_frame_count"] == 9 * CHIMERAX_ROTATION_FRAMES_PER_SCENE


def test_movie_encoding_rejects_partial_raw_frame_series(tmp_path: Path) -> None:
    frame_directory = tmp_path / "frames"
    frame_directory.mkdir()
    Image.new("RGB", (2, 2), "white").save(frame_directory / "frame-00001.png")

    with pytest.raises(RuntimeError, match="Expected 2 ChimeraX frames, found 1"):
        movie_runtime.encode_movie_frames(
            frame_directory=frame_directory,
            movie_path=tmp_path / "movie.mp4",
            log_path=tmp_path / "render.log",
            frame_width=2,
            frame_height=2,
            frames_per_scene=1,
            hold_frames_per_scene=1,
            frame_rate=24,
            expected_raw_frame_count=2,
        )


def test_chimerax_movie_frame_flattening_preserves_opaque_black_and_whitens_transparency(tmp_path: Path) -> None:
    frame_path = tmp_path / "frame.png"
    frame = Image.new("RGBA", (2, 2), (0, 0, 0, 0))
    frame.putpixel((0, 0), (0, 0, 0, 255))
    frame.putpixel((1, 0), (196, 76, 90, 255))
    frame.save(frame_path)

    flattened = Image.frombytes(
        "RGB",
        (2, 2),
        movie_runtime.flatten_movie_frame(frame_path, width=2, height=2),
    )

    assert flattened.getpixel((0, 0)) == (0, 0, 0)
    assert flattened.getpixel((1, 0)) == (196, 76, 90)
    assert flattened.getpixel((0, 1)) == (255, 255, 255)


def test_chimerax_movie_frame_validation_rejects_uncleared_corners(tmp_path: Path) -> None:
    white = bytes([255, 255, 255] * 4)
    movie_runtime.validate_white_frame_corners(white, width=2, height=2, path=tmp_path / "white.png")

    black_corner = bytearray(white)
    black_corner[:3] = bytes([0, 0, 0])
    with pytest.raises(RuntimeError, match="background was not cleared"):
        movie_runtime.validate_white_frame_corners(
            bytes(black_corner),
            width=2,
            height=2,
            path=tmp_path / "black-corner.png",
        )
