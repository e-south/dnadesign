"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/communication_visuals/catalog.py

Stable identities for communication-facing Eco1 review artifacts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

COMMUNICATION_VISUALS_DIR_NAME = "communication_visuals"
COMMUNICATION_ROLE = "communication_facing"

MOVIE_TARGET_PROTECTED_EVIDENCE = "protected-evidence"
MOVIE_TARGET_PROPOSAL_BACKBONES = "proposal-backbones"
MOVIE_TARGET_SELECTED_ELECTROSTATICS = "selected-electrostatics"
COMMUNICATION_MOVIE_TARGETS = (
    MOVIE_TARGET_PROTECTED_EVIDENCE,
    MOVIE_TARGET_PROPOSAL_BACKBONES,
    MOVIE_TARGET_SELECTED_ELECTROSTATICS,
)

DESIGN_SPACE_MAP_ID = "communication_design_space_map"
STRUCTURE_STORY_BROWSER_ID = "communication_structure_story_browser"
STRUCTURAL_SCREEN_ID = "communication_structural_screen"
SELECTED_PANEL_ID = "communication_selected_panel"
STRUCTURE_STORY_SCRIPT_ID = "communication_structure_story_script"
STRUCTURE_STORY_MOVIE_ID = "communication_structure_story_movie"
PROPOSAL_BACKBONE_CYCLE_SCRIPT_ID = "communication_proposal_backbone_cycle_script"
PROPOSAL_BACKBONE_CYCLE_MOVIE_ID = "communication_proposal_backbone_cycle_movie"
SELECTED_ELECTROSTATIC_CYCLE_SCRIPT_ID = "communication_selected_electrostatic_cycle_script"
SELECTED_ELECTROSTATIC_CYCLE_MOVIE_ID = "communication_selected_electrostatic_cycle_movie"

STRUCTURE_STORY_SCRIPT_FILE_NAME = "eco1_structure_story.cxc"
STRUCTURE_STORY_MOVIE_FILE_NAME = "eco1_structure_story.mp4"
STRUCTURE_STORY_RENDER_MANIFEST_FILE_NAME = "eco1_structure_story_render_manifest.yaml"
STRUCTURE_STORY_LOG_FILE_NAME = "eco1_structure_story_chimerax.log"
STRUCTURE_STORY_FRAME_DIRECTORY_NAME = ".eco1_structure_story_frames"
PROPOSAL_BACKBONE_CYCLE_SCRIPT_FILE_NAME = "eco1_proposal_backbone_cycle.cxc"
PROPOSAL_BACKBONE_CYCLE_MOVIE_FILE_NAME = "eco1_proposal_backbone_cycle.mp4"
PROPOSAL_BACKBONE_CYCLE_RENDER_MANIFEST_FILE_NAME = "eco1_proposal_backbone_cycle_render_manifest.yaml"
PROPOSAL_BACKBONE_CYCLE_LOG_FILE_NAME = "eco1_proposal_backbone_cycle_chimerax.log"
PROPOSAL_BACKBONE_CYCLE_FRAME_DIRECTORY_NAME = ".eco1_proposal_backbone_cycle_frames"
SELECTED_ELECTROSTATIC_CYCLE_SCRIPT_FILE_NAME = "eco1_selected_electrostatic_cycle.cxc"
SELECTED_ELECTROSTATIC_CYCLE_MOVIE_FILE_NAME = "eco1_selected_electrostatic_cycle.mp4"
SELECTED_ELECTROSTATIC_CYCLE_RENDER_MANIFEST_FILE_NAME = "eco1_selected_electrostatic_cycle_render_manifest.yaml"
SELECTED_ELECTROSTATIC_CYCLE_LOG_FILE_NAME = "eco1_selected_electrostatic_cycle_chimerax.log"
SELECTED_ELECTROSTATIC_CYCLE_FRAME_DIRECTORY_NAME = ".eco1_selected_electrostatic_cycle_frames"

COMMUNICATION_RUNTIME_PATH_NAMES = frozenset(
    {
        STRUCTURE_STORY_RENDER_MANIFEST_FILE_NAME,
        STRUCTURE_STORY_LOG_FILE_NAME,
        STRUCTURE_STORY_FRAME_DIRECTORY_NAME,
        PROPOSAL_BACKBONE_CYCLE_RENDER_MANIFEST_FILE_NAME,
        PROPOSAL_BACKBONE_CYCLE_LOG_FILE_NAME,
        PROPOSAL_BACKBONE_CYCLE_FRAME_DIRECTORY_NAME,
        SELECTED_ELECTROSTATIC_CYCLE_RENDER_MANIFEST_FILE_NAME,
        SELECTED_ELECTROSTATIC_CYCLE_LOG_FILE_NAME,
        SELECTED_ELECTROSTATIC_CYCLE_FRAME_DIRECTORY_NAME,
    }
)

COMMUNICATION_VISUAL_IDS = (
    DESIGN_SPACE_MAP_ID,
    STRUCTURE_STORY_BROWSER_ID,
    STRUCTURAL_SCREEN_ID,
    SELECTED_PANEL_ID,
    STRUCTURE_STORY_MOVIE_ID,
    PROPOSAL_BACKBONE_CYCLE_MOVIE_ID,
    SELECTED_ELECTROSTATIC_CYCLE_MOVIE_ID,
)


def validated_movie_targets(values: tuple[str, ...]) -> frozenset[str]:
    """Parse explicit movie targets and reject unknown or duplicate values."""

    unknown = sorted(set(values) - set(COMMUNICATION_MOVIE_TARGETS))
    if unknown:
        raise ValueError(f"Unknown communication movie target(s): {', '.join(unknown)}")
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise ValueError(f"Duplicate communication movie target(s): {', '.join(duplicates)}")
    return frozenset(values)
