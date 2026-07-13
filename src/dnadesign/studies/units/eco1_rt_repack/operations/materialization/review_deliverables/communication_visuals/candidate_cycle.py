"""ChimeraX movie comparing WT and selected ColabFold surface models."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.constants import (
    SECTION_CONSTRAINT_EVIDENCE,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.manifest import (
    file_hashes,
    make_deliverable_row,
    sha256,
)

from ..molecular_scene_contract import chimerax_reference_complex_style_commands
from .catalog import (
    CANDIDATE_CYCLE_MOVIE_FILE_NAME,
    CANDIDATE_CYCLE_MOVIE_ID,
    CANDIDATE_CYCLE_RENDER_MANIFEST_FILE_NAME,
    CANDIDATE_CYCLE_SCRIPT_FILE_NAME,
    CANDIDATE_CYCLE_SCRIPT_ID,
    COMMUNICATION_ROLE,
)
from .chimerax_story import encode_movie_frames, quoted_relative_path, run_chimerax_script
from .pose import (
    CHIMERAX_CAMERA_MATRIX,
    CHIMERAX_ELECTROSTATIC_SURFACE_TRANSPARENCY_PERCENT,
    CHIMERAX_HOLD_FRAMES_PER_SCENE,
    CHIMERAX_MOVIE_FRAME_RATE,
    CHIMERAX_MOVIE_HEIGHT,
    CHIMERAX_MOVIE_VIEW_ZOOM,
    CHIMERAX_MOVIE_WIDTH,
    CHIMERAX_ROTATION_DEGREES_PER_SCENE,
    CHIMERAX_ROTATION_FRAMES_PER_SCENE,
    CHIMERAX_VIEW_PADDING,
)
from .style import PROTEIN_SURFACE_COLOR, policy_label

_FRAME_DIRECTORY_NAME = ".eco1_selected_candidate_cycle_frames"
_LOG_FILE_NAME = "eco1_selected_candidate_cycle_chimerax.log"
_COULOMBIC_RANGE = "-10,10"
_COULOMBIC_KEY_COMMAND = (
    "key red-white-blue :-10 :0 :10 pos 0.70,0.075 size 0.25,0.05 "
    "font Arial fontSize 18 labelColor black showTool false"
)


def write_candidate_cycle(
    *,
    panel_root: Path,
    selected_rows: list[dict[str, Any]],
    selection_panel_path: Path,
    foldcheck_full_structure_set_path: Path,
    reference_structure_path: Path,
    render_chimerax: bool,
) -> list[dict[str, Any]]:
    """Write the WT-plus-selected surface script and optional movie."""

    panel_root.mkdir(parents=True, exist_ok=True)
    selected_rows = sorted(
        selected_rows,
        key=lambda row: (int(row.get("selection_rank") or 999), str(row.get("candidate_id") or "")),
    )
    structures = _selected_structure_paths(
        selected_rows=selected_rows,
        full_structure_set_path=foldcheck_full_structure_set_path,
    )
    script_path = panel_root / CANDIDATE_CYCLE_SCRIPT_FILE_NAME
    movie_path = panel_root / CANDIDATE_CYCLE_MOVIE_FILE_NAME
    frame_directory = panel_root / _FRAME_DIRECTORY_NAME
    render_manifest_path = panel_root / CANDIDATE_CYCLE_RENDER_MANIFEST_FILE_NAME
    write_candidate_cycle_script(
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
        **{f"selected_structure_{index:02d}": path for index, path in enumerate(structures, start=1)},
    }
    render_status, render_reason = materialize_candidate_cycle_movie(
        script_path=script_path,
        movie_path=movie_path,
        frame_directory=frame_directory,
        render_manifest_path=render_manifest_path,
        source_paths=source_paths,
        render_chimerax=render_chimerax,
    )
    source_tables = [
        "generation_policies_v3/selection/candidate_selection_panel.parquet",
        "generation_policies_v3/foldcheck_review/foldcheck_full_structure_set.yaml",
        "generation_policies_v3/foldcheck_review/structures/full_fold_set/*.pdb",
    ]
    script_row = make_deliverable_row(
        deliverable_id=CANDIDATE_CYCLE_SCRIPT_ID,
        section=SECTION_CONSTRAINT_EVIDENCE,
        artifact_kind="chimerax_script",
        status="rendered",
        path=script_path,
        source_tables=source_tables,
        input_hashes=file_hashes({**source_paths, "candidate_cycle_script": script_path}),
        alt_text="ChimeraX script that compares WT and selected RT Coulombic surfaces in one retained complex.",
        description=(
            "Provides reproducible alignment, surface coloring, camera, labeling, and frame-capture commands."
        ),
        interpretation_limit=(
            "The script controls presentation and does not alter candidate selection or structure metrics."
        ),
        title="ChimeraX can reproduce the selected-surface comparison",
        role="operator_review",
    )
    movie_status = render_status if movie_path.exists() else _missing_output_status(render_status)
    movie_row = make_deliverable_row(
        deliverable_id=CANDIDATE_CYCLE_MOVIE_ID,
        section=SECTION_CONSTRAINT_EVIDENCE,
        artifact_kind="video",
        status=movie_status,
        path=movie_path,
        source_tables=[*source_tables, CANDIDATE_CYCLE_SCRIPT_FILE_NAME],
        input_hashes=file_hashes({**source_paths, "candidate_cycle_script": script_path}),
        alt_text=(
            "ChimeraX movie that rotates WT Eco1 RT and each of eight selected fitted RT models through one full "
            "five-second turn. Every opaque protein surface uses the same -10 to +10 kcal/(mol e) Coulombic-"
            "potential scale at 298 K, while retained DNA and RNA remain gold and salmon backbone cartoons with "
            "matching ladder rungs."
        ),
        description=(
            "Keeps one camera, potential range, and retained nucleic-acid context while WT and selected models "
            "appear sequentially. Each label names the variant and its design group."
        ),
        interpretation_limit=(
            "Visual overlap is a structure-model comparison. It does not establish stability, activity, "
            "processivity, or strand displacement."
        ),
        title="WT and selected RT surfaces show qualitative charge context",
        role=COMMUNICATION_ROLE,
        render_mode="wide_visual",
        skip_reason="" if movie_path.exists() else render_reason,
        evidence_summary={"selected_sequence_count": len(selected_rows), "scene_count": len(selected_rows) + 1},
    )
    return [script_row, movie_row]


def write_candidate_cycle_script(
    *,
    script_path: Path,
    reference_structure_path: Path,
    selected_rows: list[dict[str, Any]],
    structure_paths: list[Path],
    frame_directory: Path,
) -> None:
    """Write one deterministic WT-plus-selected electrostatic comparison script."""

    if len(selected_rows) != len(structure_paths):
        raise ValueError("Candidate-cycle rows and structure paths must have the same length")
    root = script_path.parent
    lines = [
        "# Eco1 selected-candidate fitted-structure cycle",
        f"windowsize {CHIMERAX_MOVIE_WIDTH} {CHIMERAX_MOVIE_HEIGHT}",
        "camera ortho",
        "lighting soft",
        "graphics silhouettes true",
        "set bgColor white",
        f"open {quoted_relative_path(reference_structure_path, root)}",
        *chimerax_reference_complex_style_commands(include_protein_surface=False),
        "surface #1/A",
        "rename #1.1 protein_surface",
        f"color #1/A {PROTEIN_SURFACE_COLOR} target c",
        f"coulombic #1/A palette red-white-blue range {_COULOMBIC_RANGE} key false",
        f"transparency #1/A {CHIMERAX_ELECTROSTATIC_SURFACE_TRANSPARENCY_PERCENT} target s",
    ]
    for model_id, (row, structure_path) in enumerate(zip(selected_rows, structure_paths, strict=True), start=2):
        short_id = str(row["candidate_id"]).removeprefix("thread_candidate_")[:12]
        lines.extend(
            [
                f"open {quoted_relative_path(structure_path, root)}",
                f"rename #{model_id} selected_{short_id}",
                f"matchmaker #{model_id}/A to #1/A showAlignment false",
                f"hide #{model_id} atoms",
                f"cartoon #{model_id}/A",
                f"color #{model_id}/A {PROTEIN_SURFACE_COLOR} target c",
                f"surface #{model_id}/A",
                f"coulombic #{model_id}/A palette red-white-blue range {_COULOMBIC_RANGE} key false",
                f"transparency #{model_id}/A {CHIMERAX_ELECTROSTATIC_SURFACE_TRANSPARENCY_PERCENT} target s",
                f"hide #{model_id}/A cartoons",
                f"hide #{model_id}/A surfaces",
            ]
        )
    lines.extend(
        [
            _COULOMBIC_KEY_COMMAND,
            f"view matrix camera {CHIMERAX_CAMERA_MATRIX}",
            f"view all pad {CHIMERAX_VIEW_PADDING}",
            f"zoom {CHIMERAX_MOVIE_VIEW_ZOOM}",
        ]
    )
    frame_number = 1
    frame_number = _append_scene_frames(
        lines,
        frame_directory=frame_directory,
        script_root=root,
        first_frame_number=frame_number,
        label="WT Eco1 RT",
        subtitle="Reference sequence",
    )
    lines.extend(["hide #1/A cartoons", "hide #1/A surfaces"])
    selected_count = len(selected_rows)
    for selected_index, row in enumerate(selected_rows, start=1):
        model_id = selected_index + 1
        lines.append(f"show #{model_id}/A cartoons")
        lines.append(f"show #{model_id}/A surfaces")
        mutation_count = int(row.get("mutation_count_total") or 0)
        short_id = str(row["candidate_id"]).removeprefix("thread_candidate_")[:12]
        label = (
            f"Selected variant {selected_index}/{selected_count} | "
            f"{policy_label(str(row.get('policy_id') or ''))} {int(row.get('within_group_rank') or 0)} | {short_id}"
        )
        charge = float(row.get("nucleic_acid_facing_charge_delta") or 0.0)
        subtitle = (
            f"{mutation_count} substitutions | shell charge {charge:+g} | "
            f"ColabFold pLDDT {float(row.get('mean_plddt') or 0.0):.1f}"
        )
        frame_number = _append_scene_frames(
            lines,
            frame_directory=frame_directory,
            script_root=root,
            first_frame_number=frame_number,
            label=label,
            subtitle=subtitle,
        )
        lines.append(f"hide #{model_id}/A cartoons")
        lines.append(f"hide #{model_id}/A surfaces")
    lines.append("exit")
    script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _append_scene_frames(
    lines: list[str],
    *,
    frame_directory: Path,
    script_root: Path,
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
    angle_per_frame = CHIMERAX_ROTATION_DEGREES_PER_SCENE / CHIMERAX_ROTATION_FRAMES_PER_SCENE
    for _ in range(CHIMERAX_ROTATION_FRAMES_PER_SCENE):
        frame_path = frame_directory / f"frame-{frame_number:05d}.png"
        lines.extend(
            [
                f"turn y {angle_per_frame:.6f}",
                "wait 1",
                f"save {quoted_relative_path(frame_path, script_root)} width {CHIMERAX_MOVIE_WIDTH} "
                f"height {CHIMERAX_MOVIE_HEIGHT} supersample 1 transparentBackground false",
            ]
        )
        frame_number += 1
    return frame_number


def materialize_candidate_cycle_movie(
    *,
    script_path: Path,
    movie_path: Path,
    frame_directory: Path,
    render_manifest_path: Path,
    source_paths: dict[str, Path],
    render_chimerax: bool,
) -> tuple[str, str]:
    """Optionally render and encode the selected-candidate movie."""

    input_hashes = file_hashes({**source_paths, "candidate_cycle_script": script_path})
    if _existing_render_is_current(
        render_manifest_path=render_manifest_path,
        movie_path=movie_path,
        input_hashes=input_hashes,
    ):
        return "rendered", ""
    if not render_chimerax:
        return "skipped_optional_render_disabled", "ChimeraX communication rendering was not requested."
    movie_path.unlink(missing_ok=True)
    render_manifest_path.unlink(missing_ok=True)
    shutil.rmtree(frame_directory, ignore_errors=True)
    frame_directory.mkdir(parents=True)
    log_path = script_path.parent / _LOG_FILE_NAME
    run_status, run_reason = run_chimerax_script(script_path=script_path, log_path=log_path)
    if run_status != "completed":
        return run_status, run_reason
    try:
        frame_count = encode_movie_frames(
            frame_directory=frame_directory,
            movie_path=movie_path,
            log_path=log_path,
            frame_width=CHIMERAX_MOVIE_WIDTH,
            frame_height=CHIMERAX_MOVIE_HEIGHT,
            frames_per_scene=CHIMERAX_ROTATION_FRAMES_PER_SCENE,
            hold_frames_per_scene=CHIMERAX_HOLD_FRAMES_PER_SCENE,
            frame_rate=CHIMERAX_MOVIE_FRAME_RATE,
        )
    except (OSError, RuntimeError) as error:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\nmovie_encoding_error: {type(error).__name__}: {error}\n")
        return "errored", f"Movie encoding failed; inspect {log_path.name}."
    payload = {
        "schema_id": "eco1_rt.communication_candidate_cycle_render",
        "schema_version": 2,
        "status": "rendered",
        "input_hashes": input_hashes,
        "movie_encoding": {
            "renderer": "ChimeraX selected-surface 16:9 PNG saves",
            "encoder": "ffmpeg",
            "background": "white",
            "frame_rate": CHIMERAX_MOVIE_FRAME_RATE,
            "frame_count": frame_count,
            "width": CHIMERAX_MOVIE_WIDTH,
            "height": CHIMERAX_MOVIE_HEIGHT,
            "rotation_degrees_per_scene": CHIMERAX_ROTATION_DEGREES_PER_SCENE,
            "rotation_frames_per_scene": CHIMERAX_ROTATION_FRAMES_PER_SCENE,
            "potential_scale": {
                "minimum": -10,
                "maximum": 10,
                "units": "kcal/(mol e) at 298 K",
            },
        },
        "output": {"path": movie_path.name, "sha256": "sha256:" + sha256(movie_path)},
    }
    render_manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return "rendered", ""


def _selected_structure_paths(
    *,
    selected_rows: list[dict[str, Any]],
    full_structure_set_path: Path,
) -> list[Path]:
    if not full_structure_set_path.exists():
        raise FileNotFoundError(full_structure_set_path)
    payload = yaml.safe_load(full_structure_set_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("structures"), list):
        raise ValueError(f"Invalid foldcheck structure-set manifest: {full_structure_set_path}")
    path_by_candidate = {
        str(row.get("candidate_id") or ""): (
            full_structure_set_path.parent / str(row["local_model_artifact_path"])
        ).resolve()
        for row in payload["structures"]
        if isinstance(row, dict) and row.get("candidate_id") and row.get("local_model_artifact_path")
    }
    paths: list[Path] = []
    for row in selected_rows:
        candidate_id = str(row.get("candidate_id") or "")
        path = path_by_candidate.get(candidate_id)
        if path is None or not path.exists():
            raise FileNotFoundError(f"Selected structure is missing for {candidate_id}: {path}")
        paths.append(path)
    return paths


def _existing_render_is_current(
    *,
    render_manifest_path: Path,
    movie_path: Path,
    input_hashes: dict[str, str],
) -> bool:
    if not render_manifest_path.exists() or not movie_path.exists():
        return False
    payload = yaml.safe_load(render_manifest_path.read_text(encoding="utf-8"))
    return isinstance(payload, dict) and payload.get("input_hashes") == input_hashes


def _missing_output_status(render_status: str) -> str:
    return render_status if render_status != "rendered" else "errored"


__all__ = ["write_candidate_cycle", "write_candidate_cycle_script"]
