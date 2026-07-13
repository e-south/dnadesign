"""ChimeraX structure-story script and render tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from PIL import Image

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables import (
    materialize_review_deliverables,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.communication_visuals import (  # noqa: E501
    chimerax_story,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.fixtures import (
    write_deliverable_inputs,
)


def test_optional_chimerax_outputs_have_a_reproducible_script_but_no_dead_dropdown(tmp_path: Path) -> None:
    write_deliverable_inputs(tmp_path)
    result = materialize_review_deliverables(
        repo_root=Path.cwd(),
        output_root=tmp_path,
        render_chimerax_png=False,
    )
    manifest = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))
    rows = {str(row["deliverable_id"]): row for row in manifest["deliverables"]}
    script_row = rows["communication_structure_story_script"]
    script_path = result.manifest_path.parent / str(script_row["path"])
    script_text = script_path.read_text(encoding="utf-8")
    assert "windowsize 1280 720" in script_text
    assert "rename #1 eco1_rt_dna_rna_complex" in script_text
    assert "name protein_role #1/A" in script_text
    assert "name dna_role #1/D" in script_text
    assert "name rna_role #1/E,F" in script_text
    assert "surface #1/A" in script_text
    assert "cartoon #1/D,E,F suppressBackboneDisplay true" in script_text
    assert "cartoon style nucleic xsect oval width 1.35 thick 0.28" in script_text
    assert "cartoon tether nucleic shape cylinder sides 8 scale 0.65 opacity 1" in script_text
    assert "show #1/D,E,F atoms" in script_text
    assert "nucleotides #1/D,E,F ladder" in script_text
    assert "nucleotides #1/D,E,F atoms" not in script_text
    assert "style #1/D,E,F stick" not in script_text
    assert "color #1/D #B97700 target acf" in script_text
    assert "color #1/E,F #C84C5A target acf" in script_text
    assert f"view matrix camera {chimerax_story.CHIMERAX_CAMERA_MATRIX}" in script_text
    assert f"view all pad {chimerax_story.CHIMERAX_VIEW_PADDING}" in script_text
    assert f"zoom {chimerax_story.CHIMERAX_MOVIE_VIEW_ZOOM}" in script_text
    assert chimerax_story.CHIMERAX_ROTATION_DEGREES_PER_SCENE == 360.0
    assert chimerax_story.CHIMERAX_ROTATION_FRAMES_PER_SCENE / chimerax_story.CHIMERAX_MOVIE_FRAME_RATE == 5.0
    expected_captured_frames = 9 * chimerax_story.CHIMERAX_ROTATION_FRAMES_PER_SCENE
    assert "movie record" not in script_text
    assert "transparentBackground true" not in script_text
    assert script_text.count("turn y 3.000000") == expected_captured_frames
    assert script_text.count("supersample 1 transparentBackground false") == expected_captured_frames
    assert "frame-00001.png" in script_text
    assert f"frame-{expected_captured_frames:05d}.png" in script_text
    assert '2dlabels text "Open combined design space" xpos 0.035 ypos 0.955' in script_text
    assert script_text.count("transparency #1/A 55 target s") == 9
    assert "transparency #1/A 0 target s" not in script_text
    assert "transparency #1/A 22 target s" not in script_text
    assert "show #1/A:" in script_text
    assert " & sidechain atoms" in script_text
    assert "target a" in script_text
    assert "chain_a_backbone.pdb" not in script_text
    assert "communication_electrostatic_surface" not in rows
    assert rows["communication_structure_story_movie"]["status"] == "skipped_optional_render_disabled"
    candidate_script_path = result.manifest_path.parent / rows["communication_candidate_cycle_script"]["path"]
    candidate_script = candidate_script_path.read_text(encoding="utf-8")
    assert "windowsize 1280 720" in candidate_script
    assert "matchmaker #2/A to #1/A showAlignment false" in candidate_script
    assert "matchmaker #3/A to #1/A showAlignment false" in candidate_script
    assert "show #1/D,E,F atoms" in candidate_script
    assert "surface #1/A" in candidate_script
    assert "surface #2/A" in candidate_script
    assert "coulombic #1/A palette red-white-blue range -10,10 key false" in candidate_script
    assert "coulombic #2/A palette red-white-blue range -10,10 key false" in candidate_script
    color_key_command = (
        "key red-white-blue :-10 :0 :10 pos 0.70,0.075 size 0.25,0.05 "
        "font Arial fontSize 18 labelColor black showTool false"
    )
    assert color_key_command in candidate_script
    assert candidate_script.index(color_key_command) > candidate_script.rindex("\nopen ")
    candidate_scene_count = int(rows["communication_candidate_cycle_movie"]["evidence_summary"]["scene_count"])
    assert candidate_script.count('2dlabels text "Coulombic potential"') == candidate_scene_count
    assert candidate_script.count('2dlabels text "kcal/(mol e) at 298 K"') == candidate_scene_count
    assert "xpos 0.025 ypos 0.955 size 23" in candidate_script
    assert "xpos 0.025 ypos 0.91 size 18" in candidate_script
    assert "transparency #1/A 0 target s" in candidate_script
    assert "transparency #2/A 0 target s" in candidate_script
    assert "transparency #1/A 35 target s" not in candidate_script
    assert "WT Eco1 RT" in candidate_script
    assert "substitutions |" in candidate_script
    assert "shell charge" in candidate_script
    expected_candidate_frames = candidate_scene_count * chimerax_story.CHIMERAX_ROTATION_FRAMES_PER_SCENE
    assert candidate_script.count("turn y 3.000000") == expected_candidate_frames
    assert rows["communication_candidate_cycle_movie"]["section"] == "constraint_evidence_for_design_mask"
    assert rows["communication_candidate_cycle_movie"]["status"] == "skipped_optional_render_disabled"


def test_chimerax_communication_render_uses_graphical_cli_and_records_outputs(tmp_path: Path, monkeypatch) -> None:
    script_path = tmp_path / "story.cxc"
    reference_path = tmp_path / "reference.pdb"
    movie_path = tmp_path / "story.mp4"
    frame_directory = tmp_path / "frames"
    render_manifest_path = tmp_path / "render.yaml"
    script_path.write_text("exit\n", encoding="utf-8")
    reference_path.write_text("END\n", encoding="utf-8")
    observed: dict[str, object] = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout="rendered", stderr="")

    def fake_encode_movie_frames(**kwargs):
        assert kwargs["frame_directory"] == frame_directory
        assert kwargs["frames_per_scene"] == chimerax_story.CHIMERAX_ROTATION_FRAMES_PER_SCENE
        assert kwargs["frame_width"] == 1280
        assert kwargs["frame_height"] == 720
        kwargs["movie_path"].write_bytes(b"mp4")
        return 9 * (chimerax_story.CHIMERAX_ROTATION_FRAMES_PER_SCENE + chimerax_story.CHIMERAX_HOLD_FRAMES_PER_SCENE)

    monkeypatch.setattr(chimerax_story, "find_chimerax", lambda: "/Applications/ChimeraX.app/ChimeraX")
    monkeypatch.setattr(chimerax_story.subprocess, "run", fake_run)
    monkeypatch.setattr(chimerax_story, "encode_movie_frames", fake_encode_movie_frames)

    status, reason = chimerax_story.materialize_chimerax_outputs(
        script_path=script_path,
        reference_structure_path=reference_path,
        movie_path=movie_path,
        frame_directory=frame_directory,
        render_manifest_path=render_manifest_path,
        render_chimerax=True,
    )

    assert status == "rendered"
    assert reason == ""
    assert observed["command"] == ["/Applications/ChimeraX.app/ChimeraX", "--script", str(script_path)]
    render_payload = yaml.safe_load(render_manifest_path.read_text(encoding="utf-8"))
    assert render_payload["movie_encoding"]["frame_count"] == 9 * (
        chimerax_story.CHIMERAX_ROTATION_FRAMES_PER_SCENE + chimerax_story.CHIMERAX_HOLD_FRAMES_PER_SCENE
    )
    assert render_payload["movie_encoding"]["renderer"] == "ChimeraX 16:9 PNG saves"
    assert render_payload["movie_encoding"]["width"] == 1280
    assert render_payload["movie_encoding"]["height"] == 720


def test_chimerax_movie_frame_flattening_preserves_opaque_black_and_whitens_transparency(tmp_path: Path) -> None:
    frame_path = tmp_path / "frame.png"
    frame = Image.new("RGBA", (2, 2), (0, 0, 0, 0))
    frame.putpixel((0, 0), (0, 0, 0, 255))
    frame.putpixel((1, 0), (196, 76, 90, 255))
    frame.save(frame_path)

    flattened = Image.frombytes(
        "RGB",
        (2, 2),
        chimerax_story.flatten_movie_frame(frame_path, width=2, height=2),
    )

    assert flattened.getpixel((0, 0)) == (0, 0, 0)
    assert flattened.getpixel((1, 0)) == (196, 76, 90)
    assert flattened.getpixel((0, 1)) == (255, 255, 255)


def test_chimerax_movie_frame_validation_rejects_uncleared_corners(tmp_path: Path) -> None:
    white = bytes([255, 255, 255] * 4)
    chimerax_story.validate_white_frame_corners(white, width=2, height=2, path=tmp_path / "white.png")

    black_corner = bytearray(white)
    black_corner[:3] = bytes([0, 0, 0])
    with pytest.raises(RuntimeError, match="background was not cleared"):
        chimerax_story.validate_white_frame_corners(
            bytes(black_corner),
            width=2,
            height=2,
            path=tmp_path / "black-corner.png",
        )
