"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/communication_visuals/chimerax_story.py

ChimeraX script generation and optional render materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..molecular_scene_contract import chimerax_reference_complex_style_commands
from .movie_runtime import quoted_absolute_path
from .pose import (
    CHIMERAX_CAMERA_MATRIX,
    CHIMERAX_HIGHLIGHT_SIDECHAIN_STICK_RADIUS,
    CHIMERAX_HIGHLIGHT_SURFACE_TRANSPARENCY_PERCENT,
    CHIMERAX_MOVIE_HEIGHT,
    CHIMERAX_MOVIE_VIEW_ZOOM,
    CHIMERAX_MOVIE_WIDTH,
    CHIMERAX_PROTECTED_SURFACE_TRANSPARENCY_PERCENT,
    CHIMERAX_ROTATION_DEGREES_PER_CAPTURE_STEP,
    CHIMERAX_ROTATION_FRAMES_PER_SCENE,
    CHIMERAX_START_ORIENTATION_OFFSET_DEGREES,
    CHIMERAX_VIEW_PADDING,
)
from .style import DNA_COLOR, PROTEIN_SURFACE_COLOR, PROTEIN_SURFACE_OPACITY, RNA_COLOR


def write_chimerax_script(
    *,
    script_path: Path,
    reference_structure_path: Path,
    frame_directory: Path,
    reference_number_by_canonical: dict[int, int],
    scene_specs: tuple[dict[str, Any], ...],
) -> None:
    """Write the deterministic protected-evidence rotation-movie script."""

    script_path.parent.mkdir(parents=True, exist_ok=True)
    reference_absolute = quoted_absolute_path(reference_structure_path)
    lines = [
        "# Eco1 RT scientific-communication structure story",
        "# Generated from the active policy-position manifest and approved communication pose.",
        f"# Visual contract: DNA {DNA_COLOR}; RNA {RNA_COLOR}; ladder nucleotides; surface alpha "
        f"{PROTEIN_SURFACE_OPACITY:.2f}.",
        f"windowsize {CHIMERAX_MOVIE_WIDTH} {CHIMERAX_MOVIE_HEIGHT}",
        "camera ortho",
        "lighting soft",
        "graphics silhouettes true",
        "set bgColor white",
        f"open {reference_absolute}",
        *chimerax_reference_complex_style_commands(include_protein_surface=False),
        "surface #1/A",
        "rename #1.1 protein_surface",
        f"color #1/A {PROTEIN_SURFACE_COLOR} target s",
        "color #1/A #8C959F target c",
        f"view matrix camera {CHIMERAX_CAMERA_MATRIX}",
        f"view all pad {CHIMERAX_VIEW_PADDING}",
        f"zoom {CHIMERAX_MOVIE_VIEW_ZOOM}",
        f"turn y {CHIMERAX_START_ORIENTATION_OFFSET_DEGREES:.6f}",
        "set bgColor white",
    ]
    frame_number = 1
    for scene in scene_specs:
        scene_commands, frame_number = _chimerax_scene_commands(
            scene,
            reference_number_by_canonical=reference_number_by_canonical,
            frame_directory=frame_directory,
            first_frame_number=frame_number,
        )
        lines.extend(scene_commands)
    lines.append("exit")
    script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _chimerax_scene_commands(
    scene: dict[str, Any],
    *,
    reference_number_by_canonical: dict[int, int],
    frame_directory: Path,
    first_frame_number: int,
) -> tuple[list[str], int]:
    residue_numbers = sorted(
        reference_number_by_canonical[position]
        for position in scene["positions"]
        if position in reference_number_by_canonical
    )
    commands = [
        f"color #1/A {PROTEIN_SURFACE_COLOR} target s",
        "hide #1/A atoms",
        "2dlabels delete all",
        "set bgColor white",
        f'2dlabels text "{scene["label"]}" xpos 0.035 ypos 0.955 size 25 color black bgColor white',
        f"transparency #1/A {CHIMERAX_PROTECTED_SURFACE_TRANSPARENCY_PERCENT} target s",
    ]
    if residue_numbers:
        residue_selection = f"#1/A:{_residue_ranges(residue_numbers)}"
        sidechain_selection = f"{residue_selection} & sidechain"
        commands.extend(
            [
                f"color {residue_selection} {scene['color']} target s",
                f"transparency {residue_selection} {CHIMERAX_HIGHLIGHT_SURFACE_TRANSPARENCY_PERCENT} target s",
                f"color {residue_selection} {scene['color']} target c",
                f"show {sidechain_selection} atoms",
                f"style {sidechain_selection} stick",
                f"size {sidechain_selection} stickRadius {CHIMERAX_HIGHLIGHT_SIDECHAIN_STICK_RADIUS:.2f}",
                f"color {sidechain_selection} {scene['color']} target a",
            ]
        )
    next_frame_number = first_frame_number
    for capture_index in range(CHIMERAX_ROTATION_FRAMES_PER_SCENE):
        frame_path = frame_directory / f"frame-{next_frame_number:05d}.png"
        frame_absolute = quoted_absolute_path(frame_path)
        if capture_index:
            commands.append(f"turn y {CHIMERAX_ROTATION_DEGREES_PER_CAPTURE_STEP:.6f}")
        commands.extend(
            [
                "wait 1",
                f"save {frame_absolute} width {CHIMERAX_MOVIE_WIDTH} height {CHIMERAX_MOVIE_HEIGHT} "
                "supersample 1 transparentBackground false",
            ]
        )
        next_frame_number += 1
    return commands, next_frame_number


def _residue_ranges(values: list[int]) -> str:
    runs: list[str] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        runs.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    runs.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(runs)
