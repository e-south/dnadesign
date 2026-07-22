"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/test_structure_story_chimerax.py

ChimeraX structure-story script and render tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables import (
    materialize_review_deliverables,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.communication_visuals import (  # noqa: E501
    chimerax_story,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.communication_visuals.pose import (  # noqa: E501
    CHIMERAX_MOVIE_FRAME_RATE,
    CHIMERAX_ROTATION_DEGREES_PER_CAPTURE_STEP,
    CHIMERAX_ROTATION_DEGREES_PER_SCENE,
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
    first_frame_index = script_text.index("frame-00001.png")
    assert script_text.index("turn y 180.000000") < first_frame_index
    assert CHIMERAX_ROTATION_DEGREES_PER_SCENE == 360.0
    assert chimerax_story.CHIMERAX_ROTATION_FRAMES_PER_SCENE / CHIMERAX_MOVIE_FRAME_RATE == 5.0
    expected_scene_count = 10
    expected_captured_frames = expected_scene_count * chimerax_story.CHIMERAX_ROTATION_FRAMES_PER_SCENE
    assert "movie record" not in script_text
    assert "transparentBackground true" not in script_text
    expected_rotation_steps = expected_scene_count * (chimerax_story.CHIMERAX_ROTATION_FRAMES_PER_SCENE - 1)
    assert script_text.count(f"turn y {CHIMERAX_ROTATION_DEGREES_PER_CAPTURE_STEP:.6f}") == expected_rotation_steps
    assert script_text.count("supersample 1 transparentBackground false") == expected_captured_frames
    assert "frame-00001.png" in script_text
    assert f"frame-{expected_captured_frames:05d}.png" in script_text
    assert '2dlabels text "Open combined design space" xpos 0.035 ypos 0.955' in script_text
    assert script_text.count("transparency #1/A 55 target s") == expected_scene_count
    highlighted_surface_commands = re.findall(r"transparency #1/A:[^\n]+ 8 target s", script_text)
    highlighted_cartoon_commands = re.findall(r"color #1/A:[^\n]+ #[0-9A-F]{6} target c", script_text)
    assert len(highlighted_surface_commands) == expected_scene_count - 1
    assert len(highlighted_cartoon_commands) == expected_scene_count - 1
    assert script_text.count("stickRadius 0.32") == expected_scene_count - 1
    first_highlight_label = script_text.index('2dlabels text "NAxxH, YADD, and VTG context windows"')
    first_highlight_frame = script_text.index("frame-00121.png")
    first_highlight_commands = script_text[first_highlight_label:first_highlight_frame]
    background_transparency_index = first_highlight_commands.index("transparency #1/A 55 target s")
    highlighted_surface_color_index = first_highlight_commands.index("color #1/A:")
    highlighted_surface_transparency_index = first_highlight_commands.index(" 8 target s")
    highlighted_cartoon_color_index = first_highlight_commands.index("target c")
    assert (
        background_transparency_index
        < highlighted_surface_color_index
        < highlighted_surface_transparency_index
        < highlighted_cartoon_color_index
    )
    assert '2dlabels text "Wang thumb-contact track" xpos 0.035 ypos 0.955' in script_text
    assert "transparency #1/A 0 target s" not in script_text
    assert "transparency #1/A 22 target s" not in script_text
    assert "show #1/A:" in script_text
    assert " & sidechain atoms" in script_text
    assert "target a" in script_text
    assert "chain_a_backbone.pdb" not in script_text
    assert "communication_electrostatic_surface" not in rows
    assert rows["communication_structure_story_movie"]["status"] == "skipped_optional_render_disabled"
    proposal_script_path = result.manifest_path.parent / rows["communication_proposal_backbone_cycle_script"]["path"]
    proposal_script = proposal_script_path.read_text(encoding="utf-8")
    assert proposal_script.count("align #1001/A:3-311@CA toAtoms #1000/A:1-309@CA") == 2
    assert "#2000" not in proposal_script
    assert "cutoffDistance none" not in proposal_script
    assert proposal_script.count("hide #1001/A:1-2,312-320 cartoons") == 2
    assert proposal_script.count("close #1001") == 2
    assert "matchmaker" not in proposal_script
    assert "surface" not in proposal_script
    assert rows["communication_proposal_backbone_cycle_movie"]["section"] == "proteinmpnn_designs_and_fold_triage"
    assert rows["communication_proposal_backbone_cycle_movie"]["status"] == "skipped_optional_render_disabled"
    assert rows["communication_proposal_backbone_cycle_movie"]["evidence_summary"]["source_candidate_count"] == 2
    assert rows["communication_proposal_backbone_cycle_movie"]["evidence_summary"]["rendered_candidate_count"] == 2

    candidate_script_path = (
        result.manifest_path.parent / rows["communication_selected_electrostatic_cycle_script"]["path"]
    )
    candidate_script = candidate_script_path.read_text(encoding="utf-8")
    candidate_first_frame_index = candidate_script.index("frame-00001.png")
    assert candidate_script.index("turn y 180.000000") < candidate_first_frame_index
    candidate_script = candidate_script_path.read_text(encoding="utf-8")
    assert "windowsize 1280 720" in candidate_script
    assert candidate_script.count("align #1001/A:3-311@CA toAtoms #1/A:1-309@CA") == 2
    assert "cutoffDistance none" not in candidate_script
    assert candidate_script.count("hide #1001/A:1-2,312-320 cartoons") == 2
    assert candidate_script.count("close #1001") == 2
    assert "matchmaker" not in candidate_script
    assert "show #1/D,E,F atoms" in candidate_script
    assert "surface #1/A:1-309" in candidate_script
    assert candidate_script.count("surface #1001/A:3-311") == 2
    assert "coulombic #1/A:1-309 palette red-white-blue range -10,10 key false" in candidate_script
    assert candidate_script.count("coulombic #1001/A:3-311 palette red-white-blue range -10,10 key false") == 2
    color_key_command = (
        "key red-white-blue :-10 :0 :10 pos 0.70,0.075 size 0.25,0.05 "
        "font Arial fontSize 18 labelColor black showTool false"
    )
    assert color_key_command in candidate_script
    candidate_scene_count = int(
        rows["communication_selected_electrostatic_cycle_movie"]["evidence_summary"]["scene_count"]
    )
    assert candidate_script.count('2dlabels text "Coulombic potential"') == candidate_scene_count
    assert candidate_script.count('2dlabels text "kcal/(mol e) at 298 K"') == candidate_scene_count
    assert "xpos 0.025 ypos 0.955 size 23" in candidate_script
    assert "xpos 0.025 ypos 0.91 size 18" in candidate_script
    assert "transparency #1/A:1-309 0 target s" in candidate_script
    assert candidate_script.count("transparency #1001/A:3-311 0 target s") == 2
    assert "transparency #1/A 35 target s" not in candidate_script
    assert "Ec86 RT reference structure" in candidate_script
    assert "substitutions |" in candidate_script
    assert "5-10 A charge-class delta" in candidate_script
    assert "shell charge" not in candidate_script
    expected_candidate_rotation_steps = candidate_scene_count * (chimerax_story.CHIMERAX_ROTATION_FRAMES_PER_SCENE - 1)
    assert (
        candidate_script.count(f"turn y {CHIMERAX_ROTATION_DEGREES_PER_CAPTURE_STEP:.6f}")
        == expected_candidate_rotation_steps
    )
    assert rows["communication_selected_electrostatic_cycle_movie"]["section"] == "panel_selection"
    assert rows["communication_selected_electrostatic_cycle_movie"]["status"] == "skipped_optional_render_disabled"
