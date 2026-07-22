"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/communication_visuals/selected_electrostatic_cycle.py

Selected-panel ChimeraX electrostatic-surface comparison.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.constants import (
    SECTION_PANEL_SELECTION,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.manifest import (
    file_hashes,
    make_deliverable_row,
)

from ..molecular_scene_contract import chimerax_reference_complex_style_commands
from .catalog import (
    COMMUNICATION_ROLE,
    SELECTED_ELECTROSTATIC_CYCLE_FRAME_DIRECTORY_NAME,
    SELECTED_ELECTROSTATIC_CYCLE_LOG_FILE_NAME,
    SELECTED_ELECTROSTATIC_CYCLE_MOVIE_FILE_NAME,
    SELECTED_ELECTROSTATIC_CYCLE_MOVIE_ID,
    SELECTED_ELECTROSTATIC_CYCLE_RENDER_MANIFEST_FILE_NAME,
    SELECTED_ELECTROSTATIC_CYCLE_SCRIPT_FILE_NAME,
    SELECTED_ELECTROSTATIC_CYCLE_SCRIPT_ID,
)
from .movie_runtime import MovieRenderSpec, materialize_chimerax_movie, quoted_absolute_path
from .pose import (
    CHIMERAX_CAMERA_MATRIX,
    CHIMERAX_ELECTROSTATIC_SURFACE_TRANSPARENCY_PERCENT,
    CHIMERAX_HOLD_FRAMES_PER_SCENE,
    CHIMERAX_MOVIE_FRAME_RATE,
    CHIMERAX_MOVIE_HEIGHT,
    CHIMERAX_MOVIE_VIEW_ZOOM,
    CHIMERAX_MOVIE_WIDTH,
    CHIMERAX_ROTATION_DEGREES_PER_CAPTURE_STEP,
    CHIMERAX_ROTATION_DEGREES_PER_SCENE,
    CHIMERAX_ROTATION_FRAMES_PER_SCENE,
    CHIMERAX_START_ORIENTATION_OFFSET_DEGREES,
    CHIMERAX_VIEW_PADDING,
)
from .structure_set import FoldcheckStructureSet
from .style import PROTEIN_SURFACE_COLOR, policy_label

_COULOMBIC_RANGE = "-10,10"
_COULOMBIC_KEY_COMMAND = (
    "key red-white-blue :-10 :0 :10 pos 0.70,0.075 size 0.25,0.05 "
    "font Arial fontSize 18 labelColor black showTool false"
)
_RENDER_SPEC = MovieRenderSpec(
    schema_id="eco1_rt.communication_selected_electrostatic_cycle_render",
    schema_version=1,
    renderer="ChimeraX selected electrostatic-surface 16:9 PNG saves",
    output_key="selected_electrostatic_cycle_movie",
    frame_width=CHIMERAX_MOVIE_WIDTH,
    frame_height=CHIMERAX_MOVIE_HEIGHT,
    frame_rate=CHIMERAX_MOVIE_FRAME_RATE,
    frames_per_scene=CHIMERAX_ROTATION_FRAMES_PER_SCENE,
    hold_frames_per_scene=CHIMERAX_HOLD_FRAMES_PER_SCENE,
)


def write_selected_electrostatic_cycle(
    *,
    panel_root: Path,
    selected_rows: list[dict[str, Any]],
    selection_panel_path: Path,
    structure_set: FoldcheckStructureSet,
    foldcheck_full_structure_set_path: Path,
    reference_structure_path: Path,
    render_requested: bool,
) -> list[dict[str, Any]]:
    """Write the reference-plus-selected electrostatic script and optional movie."""

    panel_root.mkdir(parents=True, exist_ok=True)
    selected_rows = sorted(
        selected_rows,
        key=lambda row: (int(row.get("selection_rank") or 999), str(row.get("candidate_id") or "")),
    )
    structure_by_id = structure_set.candidate_by_id
    structures: list[Path] = []
    for row in selected_rows:
        candidate_id = str(row.get("candidate_id") or "")
        structure = structure_by_id.get(candidate_id)
        if structure is None:
            raise ValueError(f"Selected candidate is absent from the foldcheck structure set: {candidate_id}")
        structures.append(structure.path)

    script_path = panel_root / SELECTED_ELECTROSTATIC_CYCLE_SCRIPT_FILE_NAME
    movie_path = panel_root / SELECTED_ELECTROSTATIC_CYCLE_MOVIE_FILE_NAME
    frame_directory = panel_root / SELECTED_ELECTROSTATIC_CYCLE_FRAME_DIRECTORY_NAME
    render_manifest_path = panel_root / SELECTED_ELECTROSTATIC_CYCLE_RENDER_MANIFEST_FILE_NAME
    write_selected_electrostatic_cycle_script(
        script_path=script_path,
        reference_structure_path=reference_structure_path,
        selected_rows=selected_rows,
        structure_paths=structures,
        frame_directory=frame_directory,
    )
    source_paths = {
        "reference_structure": reference_structure_path,
        "candidate_selection_panel": selection_panel_path,
        "foldcheck_full_structure_set": foldcheck_full_structure_set_path,
    }
    render_status, render_reason = materialize_chimerax_movie(
        script_path=script_path,
        movie_path=movie_path,
        frame_directory=frame_directory,
        render_manifest_path=render_manifest_path,
        log_path=panel_root / SELECTED_ELECTROSTATIC_CYCLE_LOG_FILE_NAME,
        source_paths=source_paths,
        render_requested=render_requested,
        spec=_RENDER_SPEC,
        expected_raw_frame_count=(len(selected_rows) + 1) * CHIMERAX_ROTATION_FRAMES_PER_SCENE,
        encoding_metadata={
            "rotation_degrees_per_scene": CHIMERAX_ROTATION_DEGREES_PER_SCENE,
            "starting_orientation_offset_degrees": CHIMERAX_START_ORIENTATION_OFFSET_DEGREES,
            "potential_scale": {
                "minimum": -10,
                "maximum": 10,
                "units": "kcal/(mol e) at 298 K",
            },
        },
    )
    input_hashes = file_hashes({**source_paths, "movie_script": script_path})
    source_tables = [
        "generation_policies_v3/selection/candidate_selection_panel.parquet",
        "generation_policies_v3/foldcheck_review/foldcheck_full_structure_set.yaml",
        "generation_policies_v3/foldcheck_review/structures/full_fold_set/*.pdb",
    ]
    script_row = make_deliverable_row(
        deliverable_id=SELECTED_ELECTROSTATIC_CYCLE_SCRIPT_ID,
        section=SECTION_PANEL_SELECTION,
        artifact_kind="chimerax_script",
        status="rendered",
        path=script_path,
        source_tables=source_tables,
        input_hashes=input_hashes,
        alt_text="ChimeraX script for qualitative Coulombic surfaces of the reference and selected RT panel.",
        description="Fixes the residue mapping, camera, potential range, labels, and frame-capture sequence.",
        interpretation_limit="The script controls presentation and does not alter selection or structure metrics.",
        title="Selected-panel Coulombic-potential movie script",
        role="operator_review",
    )
    movie_row = make_deliverable_row(
        deliverable_id=SELECTED_ELECTROSTATIC_CYCLE_MOVIE_ID,
        section=SECTION_PANEL_SELECTION,
        artifact_kind="video",
        status=render_status if movie_path.exists() else _missing_output_status(render_status),
        path=movie_path,
        source_tables=[*source_tables, SELECTED_ELECTROSTATIC_CYCLE_SCRIPT_FILE_NAME],
        input_hashes=input_hashes,
        alt_text=(
            "ChimeraX movie rotating the retained Ec86 RT-DNA-RNA reference and each selected fitted RT model "
            "through one full turn. Opaque protein surfaces use the same -10 to +10 kcal/(mol e) Coulombic-"
            "potential scale at 298 K; DNA and RNA remain gold and salmon ladder cartoons."
        ),
        description=(
            "Uses one camera, potential range, and retained nucleic-acid context while the reference and selected "
            "models appear sequentially. Each candidate is aligned over the same 309 mapped C-alpha atoms."
        ),
        interpretation_limit=(
            "Coulombic-potential coloring is a qualitative charge-context comparison. It does not establish "
            "binding, activity, processivity, or strand displacement."
        ),
        title="Selected RT models show qualitative Coulombic surface contrasts",
        role=COMMUNICATION_ROLE,
        render_mode="wide_visual",
        skip_reason="" if movie_path.exists() else render_reason,
        evidence_summary={"selected_sequence_count": len(selected_rows), "scene_count": len(selected_rows) + 1},
    )
    return [script_row, movie_row]


def write_selected_electrostatic_cycle_script(
    *,
    script_path: Path,
    reference_structure_path: Path,
    selected_rows: list[dict[str, Any]],
    structure_paths: list[Path],
    frame_directory: Path,
) -> None:
    """Write one bounded-memory selected-panel electrostatic comparison script."""

    if len(selected_rows) != len(structure_paths):
        raise ValueError("Selected electrostatic rows and structure paths must have the same length")
    lines = [
        "# Eco1 selected-panel electrostatic-surface cycle",
        f"windowsize {CHIMERAX_MOVIE_WIDTH} {CHIMERAX_MOVIE_HEIGHT}",
        "camera ortho",
        "lighting soft",
        "graphics silhouettes true",
        "set bgColor white",
        f"open {quoted_absolute_path(reference_structure_path)}",
        *chimerax_reference_complex_style_commands(include_protein_surface=False),
        "surface #1/A:1-309",
        f"color #1/A:1-309 {PROTEIN_SURFACE_COLOR} target c",
        f"coulombic #1/A:1-309 palette red-white-blue range {_COULOMBIC_RANGE} key false",
        f"transparency #1/A:1-309 {CHIMERAX_ELECTROSTATIC_SURFACE_TRANSPARENCY_PERCENT} target s",
        _COULOMBIC_KEY_COMMAND,
        f"view matrix camera {CHIMERAX_CAMERA_MATRIX}",
        f"view all pad {CHIMERAX_VIEW_PADDING}",
        f"zoom {CHIMERAX_MOVIE_VIEW_ZOOM}",
        f"turn y {CHIMERAX_START_ORIENTATION_OFFSET_DEGREES:.6f}",
    ]
    frame_number = _append_scene_frames(
        lines,
        frame_directory=frame_directory,
        first_frame_number=1,
        label="Ec86 RT reference structure",
        subtitle="Observed RT-DNA-RNA complex",
    )
    lines.extend(["hide #1/A cartoons", "hide #1/A surfaces"])
    selected_count = len(selected_rows)
    for selected_index, (row, structure_path) in enumerate(
        zip(selected_rows, structure_paths, strict=True),
        start=1,
    ):
        short_id = str(row["candidate_id"]).removeprefix("thread_candidate_")[:12]
        lines.extend(
            [
                f"open {quoted_absolute_path(structure_path)} id #1001 name selected_rt_model",
                "# ChimeraX align uses every listed atom pair when cutoffDistance is omitted.",
                "align #1001/A:3-311@CA toAtoms #1/A:1-309@CA",
                "hide #1001 atoms",
                "cartoon #1001/A:3-311",
                "hide #1001/A:1-2,312-320 cartoons",
                f"color #1001/A:3-311 {PROTEIN_SURFACE_COLOR} target c",
                "surface #1001/A:3-311",
                f"coulombic #1001/A:3-311 palette red-white-blue range {_COULOMBIC_RANGE} key false",
                (f"transparency #1001/A:3-311 {CHIMERAX_ELECTROSTATIC_SURFACE_TRANSPARENCY_PERCENT} target s"),
            ]
        )
        mutation_count = int(row.get("mutation_count_total") or 0)
        label = (
            f"Selected variant {selected_index}/{selected_count} | "
            f"{policy_label(str(row.get('policy_id') or ''))} {int(row.get('within_group_rank') or 0)} | {short_id}"
        )
        charge = float(row.get("nucleic_acid_facing_charge_delta") or 0.0)
        subtitle = (
            f"{mutation_count} substitutions | 5-10 A charge-class delta {charge:+g} | "
            f"ColabFold pLDDT {float(row.get('mean_plddt') or 0.0):.1f}"
        )
        frame_number = _append_scene_frames(
            lines,
            frame_directory=frame_directory,
            first_frame_number=frame_number,
            label=label,
            subtitle=subtitle,
        )
        lines.append("close #1001")
    lines.append("exit")
    script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _append_scene_frames(
    lines: list[str],
    *,
    frame_directory: Path,
    first_frame_number: int,
    label: str,
    subtitle: str,
) -> int:
    lines.extend(
        [
            "2dlabels delete all",
            f'2dlabels text "{label}" xpos 0.025 ypos 0.955 size 23 color black bgColor white',
            f'2dlabels text "{subtitle}" xpos 0.025 ypos 0.91 size 18 color #57606A bgColor white',
            '2dlabels text "Coulombic potential" xpos 0.70 ypos 0.19 size 18 color black bgColor white',
            '2dlabels text "kcal/(mol e) at 298 K" xpos 0.70 ypos 0.155 size 15 color #57606A bgColor white',
        ]
    )
    frame_number = first_frame_number
    for capture_index in range(CHIMERAX_ROTATION_FRAMES_PER_SCENE):
        frame_path = frame_directory / f"frame-{frame_number:05d}.png"
        if capture_index:
            lines.append(f"turn y {CHIMERAX_ROTATION_DEGREES_PER_CAPTURE_STEP:.6f}")
        lines.extend(
            [
                "wait 1",
                f"save {quoted_absolute_path(frame_path)} width {CHIMERAX_MOVIE_WIDTH} "
                f"height {CHIMERAX_MOVIE_HEIGHT} supersample 1 transparentBackground false",
            ]
        )
        frame_number += 1
    return frame_number


def _missing_output_status(render_status: str) -> str:
    return render_status if render_status != "rendered" else "errored"


__all__ = ["write_selected_electrostatic_cycle", "write_selected_electrostatic_cycle_script"]
