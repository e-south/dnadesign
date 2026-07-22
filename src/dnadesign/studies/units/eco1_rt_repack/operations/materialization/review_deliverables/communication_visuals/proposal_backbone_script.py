"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/communication_visuals/proposal_backbone_script.py

ChimeraX script rendering for the Eco1 proposal-backbone cycle.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.generation_policies.constants import (
    COMBINED_NEAR_PLUS_DISTAL_POLICY_ID,
    DISTAL_SCAFFOLD_POLICY_ID,
    NEAR_DNA_RNA_ACID_FREE_POLICY_ID,
)

from .movie_runtime import quoted_absolute_path
from .pose import (
    CHIMERAX_CAMERA_MATRIX,
    CHIMERAX_MOVIE_HEIGHT,
    CHIMERAX_MOVIE_WIDTH,
    CHIMERAX_ROTATION_FRAMES_PER_SCENE,
    CHIMERAX_START_ORIENTATION_OFFSET_DEGREES,
)

REFERENCE_BACKBONE_COLOR = "#E8E4DA"
RETAINED_BACKBONE_COLOR = "#4F6270"
REFERENCE_INTRO_RAW_FRAMES = 12
PROPOSAL_ENCODED_FRAMES_PER_RAW_FRAME = 2
PROPOSAL_RAW_FRAMES_PER_FULL_ROTATION = CHIMERAX_ROTATION_FRAMES_PER_SCENE // PROPOSAL_ENCODED_FRAMES_PER_RAW_FRAME
PROPOSAL_CENTERED_VIEW_PADDING = 0.08
PROPOSAL_CENTERED_ZOOM = 1.05
PROPOSAL_VERTICAL_SHIFT_ANGSTROM = -4.0

POLICY_LABELS = {
    DISTAL_SCAFFOLD_POLICY_ID: "Distal redesign",
    NEAR_DNA_RNA_ACID_FREE_POLICY_ID: "Peripheral redesign",
    COMBINED_NEAR_PLUS_DISTAL_POLICY_ID: "Combined redesign",
}
SCENE_ORDER = (
    DISTAL_SCAFFOLD_POLICY_ID,
    NEAR_DNA_RNA_ACID_FREE_POLICY_ID,
    COMBINED_NEAR_PLUS_DISTAL_POLICY_ID,
)


@dataclass(frozen=True)
class ProposalBackboneScene:
    """One proposal frame with explicit structural-review context."""

    candidate_id: str
    structure_path: Path
    policy_id: str
    chapter_label: str
    chapter_position: int
    chapter_size: int
    mutation_count: int
    wt_sequence_identity_percent: float


@dataclass(frozen=True)
class ProposalBackboneChapter:
    """One policy chapter containing local-geometry-retained model scenes."""

    policy_id: str
    policy_label: str
    scenes: tuple[ProposalBackboneScene, ...]

    @property
    def content_frame_count(self) -> int:
        return len(self.scenes)

    @property
    def raw_frame_count(self) -> int:
        return ceil(self.content_frame_count / PROPOSAL_RAW_FRAMES_PER_FULL_ROTATION) * (
            PROPOSAL_RAW_FRAMES_PER_FULL_ROTATION
        )


def build_proposal_backbone_chapters(
    scenes: tuple[ProposalBackboneScene, ...],
) -> tuple[ProposalBackboneChapter, ...]:
    """Group retained scenes by generation policy."""

    chapters = []
    for policy_id in POLICY_LABELS:
        policy_scenes = tuple(scene for scene in scenes if scene.policy_id == policy_id)
        if policy_scenes:
            chapters.append(
                ProposalBackboneChapter(
                    policy_id=policy_id,
                    policy_label=POLICY_LABELS[policy_id],
                    scenes=policy_scenes,
                )
            )
    return tuple(chapters)


def proposal_backbone_raw_frame_count(scenes: tuple[ProposalBackboneScene, ...]) -> int:
    """Return the exact raw-frame count while ending every chapter at its starting pose."""

    chapters = build_proposal_backbone_chapters(scenes)
    return REFERENCE_INTRO_RAW_FRAMES + sum(chapter.raw_frame_count for chapter in chapters)


def write_proposal_backbone_cycle_script(
    *,
    script_path: Path,
    reference_backbone_path: Path,
    scenes: tuple[ProposalBackboneScene, ...],
    frame_directory: Path,
) -> None:
    """Write a bounded-memory centered movie of retained proposal backbones."""

    if not scenes:
        raise ValueError("Proposal-backbone movie requires at least one candidate scene")
    if not reference_backbone_path.is_file():
        raise FileNotFoundError(reference_backbone_path)
    chapters = build_proposal_backbone_chapters(scenes)
    script_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Eco1 ProteinMPNN proposal backbone cycle",
        "# Every ColabFold model is aligned over residues 3-311 to reference residues 1-309.",
        "# Only models retained by the declared local-geometry review are rendered.",
        f"windowsize {CHIMERAX_MOVIE_WIDTH} {CHIMERAX_MOVIE_HEIGHT}",
        "camera ortho",
        "lighting soft",
        "graphics silhouettes true",
        "set bgColor white",
        f"open {quoted_absolute_path(reference_backbone_path)} id #1000 name cryoem_reference",
        "hide #1000 atoms",
        "cartoon #1000/A:1-309",
        f"color #1000/A:1-309 {REFERENCE_BACKBONE_COLOR} target c",
        f"view matrix camera {CHIMERAX_CAMERA_MATRIX}",
        f"view #1000 pad {PROPOSAL_CENTERED_VIEW_PADDING}",
        f"zoom {PROPOSAL_CENTERED_ZOOM}",
        f"turn y {CHIMERAX_START_ORIENTATION_OFFSET_DEGREES:.6f} models #1000 center #1000",
        f"move y {PROPOSAL_VERTICAL_SHIFT_ANGSTROM:.1f} models #1000",
        "2dlabels delete all",
        _centered_label_command(
            "Local-geometry-retained ColabFold proposal models",
            ypos=0.955,
            size=24,
            color="black",
            bold=True,
        ),
        (
            '2dlabels text "Cryo-EM reference" xpos 0.405 ypos 0.08 size 17 color black '
            f"bgColor {REFERENCE_BACKBONE_COLOR}"
        ),
    ]
    frame_number = 1
    for _ in range(REFERENCE_INTRO_RAW_FRAMES):
        frame_number = _append_frame(
            lines,
            frame_directory=frame_directory,
            frame_number=frame_number,
            model_visible=False,
            rotation_degrees=None,
        )
    for chapter in chapters:
        frame_number = _append_chapter_frames(
            lines,
            chapter=chapter,
            frame_directory=frame_directory,
            first_frame_number=frame_number,
        )
    lines.append("exit")
    script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _append_chapter_frames(
    lines: list[str],
    *,
    chapter: ProposalBackboneChapter,
    frame_directory: Path,
    first_frame_number: int,
) -> int:
    current: ProposalBackboneScene | None = None
    frame_number = first_frame_number
    rotation_degrees = _chapter_rotation_degrees_per_step(chapter)
    for frame_index in range(chapter.raw_frame_count):
        scene = _stream_scene_for_frame(
            chapter.scenes,
            frame_index=frame_index,
            rendered_frame_count=chapter.raw_frame_count,
        )
        if scene is not current:
            _replace_stream_model(
                lines,
                current=current,
                replacement=scene,
                model_id=1001,
                reference_model_id=1000,
                model_name="retained_model",
                color=RETAINED_BACKBONE_COLOR,
            )
            current = scene
        _append_chapter_labels(lines, chapter=chapter, scene=scene)
        frame_number = _append_frame(
            lines,
            frame_directory=frame_directory,
            frame_number=frame_number,
            model_visible=scene is not None,
            rotation_degrees=None if frame_index == 0 else rotation_degrees,
        )
    if current is not None:
        lines.append("close #1001")
    return frame_number


def _stream_scene_for_frame(
    scenes: tuple[ProposalBackboneScene, ...],
    *,
    frame_index: int,
    rendered_frame_count: int,
) -> ProposalBackboneScene | None:
    if not scenes:
        return None
    scene_index = min(len(scenes) - 1, frame_index * len(scenes) // rendered_frame_count)
    return scenes[scene_index]


def _replace_stream_model(
    lines: list[str],
    *,
    current: ProposalBackboneScene | None,
    replacement: ProposalBackboneScene | None,
    model_id: int,
    reference_model_id: int,
    model_name: str,
    color: str,
) -> None:
    if current is not None:
        lines.append(f"close #{model_id}")
    if replacement is None:
        return
    lines.extend(
        [
            f"open {quoted_absolute_path(replacement.structure_path)} id #{model_id} name {model_name}",
            "# ChimeraX align uses every listed atom pair when cutoffDistance is omitted.",
            f"align #{model_id}/A:3-311@CA toAtoms #{reference_model_id}/A:1-309@CA",
            f"hide #{model_id} atoms",
            f"cartoon #{model_id}/A:3-311",
            f"hide #{model_id}/A:1-2,312-320 cartoons",
            f"color #{model_id}/A:3-311 {color} target c",
            f"show (#{model_id}/A:3-311 & sidechain) atoms",
            f"style (#{model_id}/A:3-311 & sidechain) stick",
            f"size (#{model_id}/A:3-311 & sidechain) stickRadius 0.08",
            f"color (#{model_id}/A:3-311 & sidechain) {color} target a",
        ]
    )


def _append_chapter_labels(
    lines: list[str],
    *,
    chapter: ProposalBackboneChapter,
    scene: ProposalBackboneScene | None,
) -> None:
    model_count = len(chapter.scenes)
    counter = _stream_counter(scene, empty_label="No retained models in this policy")
    sequence_summary = _sequence_summary(scene)
    model_label = "model" if model_count == 1 else "models"
    title = f"{chapter.policy_label} | {model_count} {model_label}"
    subtitle = f"{counter} | {sequence_summary}"
    lines.extend(
        [
            "2dlabels delete all",
            _centered_label_command(title, ypos=0.955, size=24, color="black", bold=True),
            _centered_label_command(
                subtitle,
                ypos=0.900,
                size=18,
                color=RETAINED_BACKBONE_COLOR,
                bold=False,
            ),
            (
                '2dlabels text "Cryo-EM reference" xpos 0.405 ypos 0.08 size 17 color black '
                f"bgColor {REFERENCE_BACKBONE_COLOR}"
            ),
        ]
    )


def _stream_counter(scene: ProposalBackboneScene | None, *, empty_label: str) -> str:
    if scene is None:
        return empty_label
    return f"Model {scene.chapter_position}/{scene.chapter_size}"


def _sequence_summary(scene: ProposalBackboneScene | None) -> str:
    if scene is None:
        return "No sequence metadata"
    substitution_label = "substitution" if scene.mutation_count == 1 else "substitutions"
    return f"WT identity {scene.wt_sequence_identity_percent:.1f}% | {scene.mutation_count} {substitution_label}"


def _centered_label_command(text: str, *, ypos: float, size: int, color: str, bold: bool) -> str:
    average_glyph_width_em = 0.46
    estimated_width_pixels = len(text) * size * average_glyph_width_em
    xpos = max(0.02, (CHIMERAX_MOVIE_WIDTH - estimated_width_pixels) / (2.0 * CHIMERAX_MOVIE_WIDTH))
    return (
        f'2dlabels text "{text}" xpos {xpos:.4f} ypos {ypos:.3f} size {size} '
        f"font Arial bold {'true' if bold else 'false'} color {color} bgColor white"
    )


def _chapter_rotation_degrees_per_step(chapter: ProposalBackboneChapter) -> float:
    completed_turns = chapter.raw_frame_count // PROPOSAL_RAW_FRAMES_PER_FULL_ROTATION
    return (360.0 * completed_turns) / (chapter.raw_frame_count - 1)


def _append_frame(
    lines: list[str],
    *,
    frame_directory: Path,
    frame_number: int,
    model_visible: bool,
    rotation_degrees: float | None,
) -> int:
    frame_path = frame_directory / f"frame-{frame_number:05d}.png"
    if rotation_degrees is not None:
        models = "#1000,1001" if model_visible else "#1000"
        lines.append(f"turn y {rotation_degrees:.6f} models {models} center #1000")
    lines.extend(
        [
            "wait 1",
            f"save {quoted_absolute_path(frame_path)} width {CHIMERAX_MOVIE_WIDTH} "
            f"height {CHIMERAX_MOVIE_HEIGHT} supersample 1 transparentBackground false",
        ]
    )
    return frame_number + 1


__all__ = [
    "POLICY_LABELS",
    "PROPOSAL_ENCODED_FRAMES_PER_RAW_FRAME",
    "PROPOSAL_RAW_FRAMES_PER_FULL_ROTATION",
    "ProposalBackboneChapter",
    "ProposalBackboneScene",
    "SCENE_ORDER",
    "build_proposal_backbone_chapters",
    "proposal_backbone_raw_frame_count",
    "write_proposal_backbone_cycle_script",
]
