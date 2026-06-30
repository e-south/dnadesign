"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/mask_tracks.py

Linear and ChimeraX mask-context deliverables for Eco1 review.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import matplotlib
from matplotlib.patches import Patch, Rectangle

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.constants import (
    SECTION_CONSTRAINT_EVIDENCE,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.manifest import (
    file_hashes,
    make_deliverable_row,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.mask_rows import (
    read_mask_residues,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.rendering import (
    LABEL_SIZE,
    LEGEND_SIZE,
    OKABE_ITO,
    TICK_SIZE,
    TITLE_SIZE,
    save_accessible_svg,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_TRACKS = (
    ("motif_protected", "Catalytic motifs", OKABE_ITO["vermillion"]),
    ("wang_ec86_direct_contact_prior", "Wang/Ec86 priors", OKABE_ITO["purple"]),
    ("direct_retained_dna_rna_contact_5a", "DNA/RNA within 5 A", OKABE_ITO["orange"]),
    ("evolutionarily_conserved_clade9_25pct_plurality", "Clade 9 plurality", OKABE_ITO["blue"]),
    ("protected", "Protected union", "#4d4d4d"),
    ("non_fixed", "Design canvas", OKABE_ITO["green"]),
    ("non_fixed_missing_backbone", "Missing-backbone canvas", "#9a9a9a"),
)
_MASK_TRACK_BLOCK_SIZE = 64
_CHIMERAX_COLOR_ORDER = (
    ("protected", "#d9d2c3", "protected union"),
    ("non_fixed", "#6aa84f", "design canvas"),
    ("evolutionarily_conserved_clade9_25pct_plurality", "#4d78a8", "clade 9 plurality"),
    ("direct_retained_dna_rna_contact_5a", "#d19a33", "5 A DNA/RNA contact"),
    ("wang_ec86_direct_contact_prior", "#8b62a8", "Wang/Ec86 prior"),
    ("motif_protected", "#bf4b4b", "motif anchor"),
)
_CHIMERAX_ORIENTATION_PRESET_ID = "ec86_reference_thumb_down_v1"
_CHIMERAX_VIEW_COMMANDS = (
    "view orient",
    "view initial",
    "turn y -78",
    "turn z -72",
    "turn x -8",
    "view",
    "zoom 1.22",
)


def write_linear_mask_tracks(
    *,
    panel_root: Path,
    mask_set_path: Path,
    mask_residues: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Render linear mask tracks over canonical residue positions."""

    residues = mask_residues if mask_residues is not None else read_mask_residues(mask_set_path)
    title = "Protected evidence defines fixed residues and the design canvas"
    residues_by_position = {int(row["canonical_position"]): row for row in residues}
    positions = sorted(residues_by_position)
    blocks = _position_blocks(positions, block_size=_MASK_TRACK_BLOCK_SIZE)
    block_count = len(blocks)
    fig_height = max(4.8, block_count * 1.76 + 1.32)
    fig, axes = plt.subplots(block_count, 1, figsize=(16.2, fig_height), squeeze=False)
    for index, (ax, block_positions) in enumerate(zip(axes.flatten(), blocks, strict=True)):
        _draw_mask_track_block(
            ax,
            block_positions=block_positions,
            residues_by_position=residues_by_position,
            panel_title=title if index == 0 else None,
        )
    fig.legend(
        handles=[Patch(facecolor=color, label=label) for _field, label, color in _TRACKS],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=len(_TRACKS),
        frameon=False,
        fontsize=LEGEND_SIZE - 1,
        columnspacing=1.05,
        handletextpad=0.45,
    )
    fig.supxlabel("Ec86 canonical residue position", fontsize=LABEL_SIZE, y=0.073)
    fig.supylabel("Mask evidence track", fontsize=LABEL_SIZE, x=0.012)
    fig.subplots_adjust(left=0.13, right=0.995, bottom=0.135, top=0.92, hspace=0.46)

    path = panel_root / "linear_mask_tracks.svg"
    alt = (
        "Linear Eco1 RT mask-track panel showing motif anchors, Wang/Ec86 priors, "
        "retained DNA/RNA 5 A contacts, clade 9 plurality protection, protected union, "
        "and mutable design canvas tracks."
    )
    save_accessible_svg(fig, path, title=title, description=alt, dpi=300)
    return make_deliverable_row(
        deliverable_id="linear_mask_tracks",
        section=SECTION_CONSTRAINT_EVIDENCE,
        artifact_kind="svg",
        status="rendered",
        path=path,
        source_tables=["mask_set.yaml"],
        input_hashes=file_hashes({"mask_set": mask_set_path}),
        alt_text=alt,
        description=(
            "Separates each protection reason and the remaining design canvas on a single residue-coordinate axis. "
            "Each square cell represents one canonical residue in one evidence track."
        ),
        interpretation_limit=(
            "Mask tracks describe what was fixed or mutable for the current ProteinMPNN "
            "run; they do not evaluate candidate quality."
        ),
        title=title,
    )


def write_mask_structure_context(
    *,
    panel_root: Path,
    mask_set_path: Path,
    reference_backbone_path: Path,
    mask_residues: list[dict[str, Any]] | None = None,
    render_png: bool = True,
) -> list[dict[str, Any]]:
    """Write a ChimeraX script and optionally a PNG if ChimeraX is available."""

    residues = mask_residues if mask_residues is not None else read_mask_residues(mask_set_path)
    script_path = panel_root / "mask_structure_context.cxc"
    orientation_template_path = panel_root / "mask_structure_context_orientation_template.cxc"
    png_path = panel_root / "mask_structure_context.png"
    _write_chimerax_script(
        script_path,
        residues=residues,
        reference_backbone_path=reference_backbone_path,
        png_path=png_path,
    )
    _write_orientation_template_script(
        orientation_template_path,
        residues=residues,
        reference_backbone_path=reference_backbone_path,
    )
    input_hashes = file_hashes({"mask_set": mask_set_path, "reference_backbone": reference_backbone_path})
    rows = [
        make_deliverable_row(
            deliverable_id="mask_structure_context_script",
            section=SECTION_CONSTRAINT_EVIDENCE,
            artifact_kind="chimerax_script",
            status="rendered",
            path=script_path,
            source_tables=["mask_set.yaml", "proteinmpnn_request/chain_a_backbone.pdb"],
            input_hashes=input_hashes,
            alt_text=(
                "ChimeraX script that colors the Ec86 RT backbone by current Eco1 "
                "mask categories on an off-white reference structure."
            ),
            description=(
                "Provides a reproducible 3D review recipe for the current mask policy "
                "without storing a large rendered scene as source."
            ),
            interpretation_limit=(
                "The script visualizes mask context only. It does not show candidate fold quality or activity."
            ),
            role="review_only",
            title="ChimeraX recipe for the Ec86 mask-context view",
        ),
        make_deliverable_row(
            deliverable_id="mask_structure_context_orientation_template",
            section=SECTION_CONSTRAINT_EVIDENCE,
            artifact_kind="chimerax_script",
            status="rendered",
            path=orientation_template_path,
            source_tables=["mask_set.yaml", "proteinmpnn_request/chain_a_backbone.pdb"],
            input_hashes=input_hashes,
            alt_text=(
                "Interactive ChimeraX script for manually tuning the Eco1 RT structure orientation "
                "before saving a reusable session view."
            ),
            description=(
                "Opens the same mask-colored Ec86 RT reference view without exiting, so the operator "
                "can rotate, pan, zoom, and save a ChimeraX session for a preferred publication pose."
            ),
            interpretation_limit=(
                "Manual orientation affects only camera/view presentation. It must not change sequence, "
                "mask, fold, or candidate evidence."
            ),
            role="operator_review",
            title="Manual ChimeraX orientation template for the Ec86 reference",
        ),
    ]
    if not render_png:
        if png_path.exists():
            status = "rendered"
            skip_reason = "Using an existing ChimeraX PNG; rendering was disabled for this materialization run."
        else:
            status = "skipped_optional_render_disabled"
            skip_reason = "ChimeraX PNG rendering was disabled for this materialization run."
    else:
        executable = _find_chimerax()
        if executable:
            chimerax_completed = _run_chimerax(executable=executable, script_path=script_path)
            status = "rendered" if chimerax_completed and png_path.exists() else "skipped_runtime_failed"
            skip_reason = (
                "" if status == "rendered" else "ChimeraX was found, but the command did not write the expected PNG."
            )
        else:
            status = "skipped_missing_runtime"
            skip_reason = "ChimeraX executable was not found on PATH or at the standard macOS app path."
    rows.append(
        make_deliverable_row(
            deliverable_id="mask_structure_context_png",
            section=SECTION_CONSTRAINT_EVIDENCE,
            artifact_kind="png",
            status=status,
            path=png_path,
            source_tables=["mask_set.yaml", "proteinmpnn_request/chain_a_backbone.pdb"],
            input_hashes=input_hashes,
            alt_text="Optional ChimeraX render of the Eco1 RT mask structure context.",
            description="Rendered ChimeraX PNG showing the Ec86 reference protein colored by mask category.",
            interpretation_limit="This render is a mask-context view, not a fold-check result.",
            role="optional_heavy",
            skip_reason=skip_reason,
            title="Ec86 reference structure maps the fixed-residue mask",
        )
    )
    return rows


def _write_chimerax_script(
    path: Path,
    *,
    residues: list[dict[str, Any]],
    reference_backbone_path: Path,
    png_path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Eco1 RT repack mask context",
        "# mask_policy_id: eco1_rt_clade9_plurality25_direct_contact5a_v1",
        f"# orientation_preset_id: {_CHIMERAX_ORIENTATION_PRESET_ID}",
        "# Paths are relative to this script directory.",
        *_chimerax_scene_setup_lines(reference_backbone_path=reference_backbone_path, script_path=path),
    ]
    lines.extend(_chimerax_mask_color_lines(residues))
    lines.extend(
        [
            *_CHIMERAX_VIEW_COMMANDS,
            '2dlabels text "Ec86 reference" xpos 0.035 ypos 0.90 size 24 color black bgColor none',
            f"save {_relative_chimerax_path(png_path, script_path=path)} width 1800 height 1200 supersample 2",
            "exit",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_orientation_template_script(
    path: Path,
    *,
    residues: list[dict[str, Any]],
    reference_backbone_path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Eco1 RT repack mask context: Manual orientation handoff",
        "# Use this script when the scripted still needs human pose tuning.",
        "# 1. Open this script in ChimeraX.",
        "# 2. Rotate, pan, and zoom until the reference orientation is correct.",
        "# 3. In the ChimeraX command line, run:",
        "#    view name eco1_publication_v1",
        "#    save mask_structure_context_orientation.cxs",
        "# 4. Send the saved .cxs session or a screenshot; the pose can then be promoted into code.",
        "# This script intentionally leaves the ChimeraX window open for manual tuning.",
        "# mask_policy_id: eco1_rt_clade9_plurality25_direct_contact5a_v1",
        f"# orientation_preset_id: {_CHIMERAX_ORIENTATION_PRESET_ID}",
        "# Paths are relative to this script directory.",
        *_chimerax_scene_setup_lines(reference_backbone_path=reference_backbone_path, script_path=path),
    ]
    lines.extend(_chimerax_mask_color_lines(residues))
    lines.extend(
        [
            *_CHIMERAX_VIEW_COMMANDS,
            '2dlabels text "Ec86 reference" xpos 0.035 ypos 0.90 size 24 color black bgColor none',
            "view name eco1_publication_v1",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _chimerax_scene_setup_lines(*, reference_backbone_path: Path, script_path: Path) -> list[str]:
    return [
        "set bgColor white",
        "camera ortho",
        "lighting soft",
        "graphics silhouettes true",
        f"open {_relative_chimerax_path(reference_backbone_path, script_path=script_path)}",
        "color #1 #efece3",
        "cartoon #1",
        "cartoon style width 1.4 thick 0.22",
    ]


def _chimerax_mask_color_lines(residues: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for field, color, label in _CHIMERAX_COLOR_ORDER:
        positions = [int(row["canonical_position"]) for row in residues if bool(row.get(field))]
        selector = _selector_for_positions(positions)
        if selector:
            lines.append(f"# {label}")
            lines.append(f"color #1/A:{selector} {color}")
    return lines


def _relative_chimerax_path(target_path: Path, *, script_path: Path) -> str:
    if not target_path.is_absolute():
        return str(target_path)
    return os.path.relpath(target_path, start=script_path.parent)


def _segments(positions: list[int]) -> list[tuple[int, int]]:
    if not positions:
        return []
    sorted_positions = sorted(set(positions))
    segments: list[tuple[int, int]] = []
    start = sorted_positions[0]
    previous = start
    for position in sorted_positions[1:]:
        if position == previous + 1:
            previous = position
            continue
        segments.append((start, previous - start + 1))
        start = previous = position
    segments.append((start, previous - start + 1))
    return segments


def _position_tick_indexes(positions: list[int]) -> list[int]:
    if not positions:
        return []
    if len(positions) <= 20:
        return list(range(len(positions)))
    indexes = [0, *range(4, len(positions), 5)]
    if indexes[-1] != len(positions) - 1:
        indexes.append(len(positions) - 1)
    return indexes


def _wt_residue_tick_indexes(positions: list[int]) -> list[int]:
    if not positions:
        return []
    return list(range(len(positions)))


def _position_blocks(positions: list[int], *, block_size: int) -> list[list[int]]:
    return [positions[start : start + block_size] for start in range(0, len(positions), block_size)]


def _draw_mask_track_block(
    ax: Any,
    *,
    block_positions: list[int],
    residues_by_position: dict[int, dict[str, Any]],
    panel_title: str | None = None,
) -> None:
    position_to_index = {position: index for index, position in enumerate(block_positions)}
    for y_index, (field, _label, color) in enumerate(_TRACKS):
        for position in block_positions:
            row = residues_by_position[position]
            x_index = position_to_index[position]
            ax.add_patch(
                Rectangle(
                    (x_index - 0.5, y_index - 0.5),
                    1.0,
                    1.0,
                    facecolor=color if bool(row.get(field)) else "#f6f8fa",
                    edgecolor="#ffffff",
                    linewidth=0.2,
                )
            )
    ax.set_xlim(-0.5, len(block_positions) - 0.5)
    ax.set_ylim(len(_TRACKS) - 0.5, -0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_yticks(range(len(_TRACKS)), [label for _field, label, _color in _TRACKS], fontsize=TICK_SIZE)
    bottom_tick_indexes = _position_tick_indexes(block_positions)
    ax.set_xticks(
        bottom_tick_indexes,
        [str(block_positions[index]) for index in bottom_tick_indexes],
        fontsize=TICK_SIZE,
    )
    top_tick_indexes = _wt_residue_tick_indexes(block_positions)
    top_axis = ax.secondary_xaxis("top")
    top_axis.set_xticks(
        top_tick_indexes,
        [str(residues_by_position[block_positions[index]].get("wt_aa") or "") for index in top_tick_indexes],
        fontsize=6.4 if len(block_positions) > 48 else TICK_SIZE,
    )
    top_axis.tick_params(length=0, pad=2)
    block_title = f"Ec86 positions {block_positions[0]}-{block_positions[-1]}"
    if panel_title:
        ax.set_title(f"{panel_title}\n{block_title}", fontsize=TITLE_SIZE, pad=22)
    else:
        ax.text(
            0.5,
            1.12,
            block_title,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=LABEL_SIZE,
        )
    ax.grid(axis="x", color="#d0d7de", alpha=0.28, linewidth=0.42)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def _selector_for_positions(positions: list[int]) -> str:
    return ",".join(
        str(start) if length == 1 else f"{start}-{start + length - 1}" for start, length in _segments(positions)
    )


def _find_chimerax() -> str:
    path_executable = shutil.which("ChimeraX") or shutil.which("chimerax")
    if path_executable:
        return path_executable
    for app_root in (Path("/Applications"), Path.home() / "Applications"):
        for app_path in sorted(app_root.glob("ChimeraX*.app")):
            executable = app_path / "Contents" / "MacOS" / "ChimeraX"
            if executable.exists():
                return str(executable)
    return ""


def _run_chimerax(*, executable: str, script_path: Path) -> bool:
    try:
        completed = subprocess.run(
            [executable, "--script", str(script_path)],
            check=False,
            cwd=script_path.parent,
            timeout=120,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0
