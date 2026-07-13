"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/foldcheck_review/structure_overlay.py

ChimeraX overlay render for selected Eco1 fold-check structures.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import matplotlib

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.foldcheck_review.models import PanelEntry
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.foldcheck_review.pdb_alignment import (
    write_aligned_pdb_to_reference_ca,
)

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

_OVERLAY_SCRIPT_NAME = "ec86_structure_overlay_panel.cxc"
_OVERLAY_IMAGE_NAME = "structure_overlay_panel.png"
_ORIENTATION_PRESET_ID = "ec86_reference_thumb_down_v1"
_VIEW_COMMANDS = (
    "view orient",
    "view initial",
    "turn y -78",
    "turn z -72",
    "turn x -8",
    "view",
    "zoom 1.18",
)
_REFERENCE_COLOR = "#d8d3c8"
_MODEL_STYLES = {
    "wild_type_runtime_baseline": ("#0072B2", 0, "WT ColabFold baseline"),
    "best_rmsd": ("#009E73", 0, "ProteinMPNN best RMSD"),
    "rmsd_outlier": ("#D55E00", 0, "ProteinMPNN RMSD outlier"),
}


@dataclass(frozen=True)
class _AlignedOverlayEntry:
    entry: PanelEntry
    aligned_path: Path
    mapped_ca_rmsd: float


def write_structure_overlay_plot_row(
    *,
    plot_root: Path,
    chimerax_root: Path,
    reference_local_path: Path,
    entries: list[PanelEntry],
    render_png: bool = False,
) -> dict[str, object]:
    """Write a ChimeraX overlay script and optionally render a PNG."""

    plot_root.mkdir(parents=True, exist_ok=True)
    chimerax_root.mkdir(parents=True, exist_ok=True)
    script_path = chimerax_root / _OVERLAY_SCRIPT_NAME
    png_path = plot_root / _OVERLAY_IMAGE_NAME
    panel_png_root = plot_root / "structure_overlay_panels"
    panel_script_root = chimerax_root / "structure_overlay_panels"
    selected_entries = _select_overlay_entries(entries)
    aligned_entries = _align_overlay_entries(
        selected_entries,
        reference_local_path=reference_local_path,
        aligned_root=chimerax_root.parent / "structures" / "overlay_aligned",
    )
    panel_scripts = _write_panel_scripts(
        panel_script_root=panel_script_root,
        panel_png_root=panel_png_root,
        reference_local_path=reference_local_path,
        entries=aligned_entries,
    )
    _write_overlay_index_script(
        script_path,
        reference_local_path=reference_local_path,
        entries=aligned_entries,
        panel_scripts=panel_scripts,
        png_path=png_path,
    )

    status = "rendered"
    skip_reason = ""
    if not reference_local_path.exists():
        status = "skipped_missing_input"
        skip_reason = f"Missing reference structure: {reference_local_path}"
    elif len(aligned_entries) < 2:
        status = "skipped_missing_input"
        skip_reason = "Fewer than two local fold-check structures are available for an overlay."
    elif not render_png:
        if png_path.exists():
            status = "rendered"
            skip_reason = "Using an existing ChimeraX PNG; rendering was disabled for this materialization run."
        else:
            status = "skipped_optional_render_disabled"
            skip_reason = "ChimeraX overlay rendering was disabled for this materialization run."
    else:
        executable = _find_chimerax()
        if not executable:
            status = "skipped_runtime_unavailable"
            skip_reason = "ChimeraX executable was not found on this machine."
        elif not _render_chimerax_panel(
            executable=executable,
            scripts=panel_scripts,
            panel_png_root=panel_png_root,
            entries=aligned_entries,
            output_path=png_path,
        ):
            status = "skipped_runtime_unavailable"
            skip_reason = "ChimeraX was found, but the separate-panel render did not write the expected PNG."

    return {
        "plot_id": "structure_overlay_panel",
        "status": status,
        "path": str(png_path),
        "title": "Reference-fitted ColabFold structures separate preserved and outlier folds",
        "alt_text": (
            "Multi-panel ChimeraX render showing selected ColabFold structures fitted to the "
            "ec86kit/7V9U reference one panel at a time."
        ),
        "description": (
            "Shows representative ColabFold structures as separate ChimeraX-fitted panels after C-alpha "
            "alignment of full-length model residues 3-311 to ec86kit/7V9U reference residues 1-309. "
            "Each panel compares one selected structure against the same reference, reports mapped C-alpha RMSD, "
            "and includes full-sequence identity plus ProteinMPNN design-position recovery when available."
        ),
        "interpretation_limit": (
            "This overlay is a visual structure-review aid. It does not measure activity, "
            "processivity, strand displacement, or hairpin readthrough."
        ),
        "data_sources": [
            "foldcheck_review/foldcheck_structure_panel.yaml",
            "foldcheck_review/foldcheck_candidate_ranking.parquet",
            "foldcheck_review/chimerax/ec86_structure_overlay_panel.cxc",
        ],
        "skip_reason": skip_reason,
    }


def _select_overlay_entries(entries: list[PanelEntry]) -> list[PanelEntry]:
    local_entries = [
        entry
        for entry in entries
        if entry.copy_status != "source_not_local" and Path(entry.local_model_artifact_path).exists()
    ]
    selected: list[PanelEntry] = []
    for stratum in ("wild_type_runtime_baseline", "best_rmsd", "rmsd_outlier"):
        match = next((entry for entry in local_entries if entry.selection_stratum == stratum), None)
        if match is not None:
            selected.append(match)
    if len(selected) >= 2:
        return selected
    for entry in local_entries:
        if entry not in selected:
            selected.append(entry)
        if len(selected) >= 3:
            break
    return selected


def _align_overlay_entries(
    entries: list[PanelEntry],
    *,
    reference_local_path: Path,
    aligned_root: Path,
) -> list[_AlignedOverlayEntry]:
    aligned_entries: list[_AlignedOverlayEntry] = []
    for entry in entries:
        local_path = Path(entry.local_model_artifact_path)
        if not local_path.exists() or not reference_local_path.exists():
            continue
        aligned_path = aligned_root / f"{entry.candidate_id}.aligned_to_ec86_reference.pdb"
        mapped_ca_rmsd = write_aligned_pdb_to_reference_ca(
            query_path=local_path,
            reference_path=reference_local_path,
            output_path=aligned_path,
        )
        aligned_entries.append(
            _AlignedOverlayEntry(entry=entry, aligned_path=aligned_path, mapped_ca_rmsd=mapped_ca_rmsd)
        )
    return aligned_entries


def _write_panel_scripts(
    *,
    panel_script_root: Path,
    panel_png_root: Path,
    reference_local_path: Path,
    entries: list[_AlignedOverlayEntry],
) -> list[Path]:
    panel_script_root.mkdir(parents=True, exist_ok=True)
    panel_png_root.mkdir(parents=True, exist_ok=True)
    scripts: list[Path] = []
    for index, aligned_entry in enumerate(entries, start=1):
        model_name = _safe_model_name(
            aligned_entry.entry.candidate_id,
            aligned_entry.entry.selection_stratum,
        )
        script_path = panel_script_root / f"{index:02d}_{model_name}.cxc"
        png_path = panel_png_root / f"{index:02d}_{aligned_entry.entry.candidate_id}.png"
        _write_single_panel_script(
            script_path,
            reference_local_path=reference_local_path,
            aligned_entry=aligned_entry,
            png_path=png_path,
        )
        scripts.append(script_path)
    return scripts


def _write_single_panel_script(
    path: Path,
    *,
    reference_local_path: Path,
    aligned_entry: _AlignedOverlayEntry,
    png_path: Path,
) -> None:
    entry = aligned_entry.entry
    color, transparency, display_label = _style_for_entry(entry)
    lines = [
        "# Eco1 fold-check selected-structure panel",
        "# Generated by dnadesign; edit a copy for manual view changes.",
        f"# orientation_preset_id: {_ORIENTATION_PRESET_ID}",
        f"# panel_label: {display_label}",
        f"# candidate_id: {entry.candidate_id}",
        f"# full_sequence_identity_percent: {_format_optional_float(entry.full_sequence_identity_percent)}",
        f"# design_position_recovery_percent: {_format_optional_float(entry.design_position_recovery_percent)}",
        f"# mapped_c_alpha_rmsd: {aligned_entry.mapped_ca_rmsd:.3f}",
        f"# wt_runtime_c_alpha_rmsd: {_format_optional_float(entry.wt_runtime_ca_rmsd)}",
        "set bgColor white",
        "camera ortho",
        "lighting soft",
        "graphics silhouettes true",
        f"open {_relative_chimerax_path(reference_local_path, script_path=path)}",
        "rename #1 ec86kit_cryoem_reference",
        "cartoon #1",
        "cartoon style width 1.35 thick 0.22",
        f"color #1 {_REFERENCE_COLOR} target r",
        "transparency #1 58 target r",
        f"open {_relative_chimerax_path(aligned_entry.aligned_path, script_path=path)}",
        f"rename #2 {_safe_model_name(entry.candidate_id, entry.selection_stratum)}",
        "cartoon #2",
        f"color #2 {color} target r",
        f"transparency #2 {transparency} target r",
        *_VIEW_COMMANDS,
        f"save {_relative_chimerax_path(png_path, script_path=path)} width 1400 height 1050 supersample 2",
        "exit",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_overlay_index_script(
    path: Path,
    *,
    reference_local_path: Path,
    entries: list[_AlignedOverlayEntry],
    panel_scripts: list[Path],
    png_path: Path,
) -> None:
    lines = [
        "# Eco1 fold-check selected-structure panel index",
        "# The rendered PNG is composed from one ChimeraX render per selected structure.",
        f"# orientation_preset_id: {_ORIENTATION_PRESET_ID}",
        f"# reference: {_relative_chimerax_path(reference_local_path, script_path=path)}",
        f"# output: {_relative_chimerax_path(png_path, script_path=path)}",
    ]
    for script in panel_scripts:
        lines.append(f"# panel_script: {_relative_chimerax_path(script, script_path=path)}")
    for aligned_entry in entries:
        entry = aligned_entry.entry
        _color, _transparency, display_label = _style_for_entry(entry)
        if entry.proteinmpnn_rank is not None:
            lines.append(f"# ProteinMPNN variant rank {entry.proteinmpnn_rank}: {entry.candidate_id}")
        lines.append(
            "# "
            f"{display_label}: {entry.candidate_id}; "
            f"full_sequence_identity_percent={_format_optional_float(entry.full_sequence_identity_percent)}; "
            f"design_position_recovery_percent={_format_optional_float(entry.design_position_recovery_percent)}; "
            f"mapped_c_alpha_rmsd={aligned_entry.mapped_ca_rmsd:.3f} A; "
            f"wt_runtime_c_alpha_rmsd={_format_optional_float(entry.wt_runtime_ca_rmsd)}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_chimerax_panel(
    *,
    executable: str,
    scripts: list[Path],
    panel_png_root: Path,
    entries: list[_AlignedOverlayEntry],
    output_path: Path,
) -> bool:
    for script in scripts:
        if not _run_chimerax(executable=executable, script_path=script):
            return False
    panel_images = sorted(panel_png_root.glob("*.png"))
    if len(panel_images) != len(entries):
        return False
    _write_combined_panel(panel_images=panel_images, entries=entries, output_path=output_path)
    return output_path.exists()


def _write_combined_panel(
    *,
    panel_images: list[Path],
    entries: list[_AlignedOverlayEntry],
    output_path: Path,
) -> None:
    column_count = len(panel_images)
    fig, axes = plt.subplots(1, column_count, figsize=(5.15 * column_count, 6.15), squeeze=False)
    for ax, image_path, aligned_entry in zip(axes.flatten(), panel_images, entries, strict=True):
        ax.imshow(mpimg.imread(image_path))
        ax.set_axis_off()
        ax.set_title(_panel_title(aligned_entry), fontsize=13.5, pad=6)
    fig.suptitle(
        "Reference-fitted ColabFold structures show representative agreement and outliers",
        fontsize=16,
        y=0.965,
    )
    legend_handles = _overlay_legend_handles(entries)
    fig.legend(
        handles=legend_handles,
        frameon=False,
        fontsize=12,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.032),
        ncol=min(4, len(legend_handles)),
    )
    fig.text(
        0.5,
        0.122,
        "Each subtitle reports mapped C-alpha RMSD after fitting the ColabFold model to the ec86kit/7V9U reference.",
        ha="center",
        va="center",
        fontsize=11.5,
    )
    fig.subplots_adjust(left=0.012, right=0.988, bottom=0.185, top=0.79, wspace=0.025)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="png", dpi=180, facecolor="white")
    plt.close(fig)


def _panel_title(aligned_entry: _AlignedOverlayEntry) -> str:
    entry = aligned_entry.entry
    _color, _transparency, display_label = _style_for_entry(entry)
    identity = _identity_title_text(entry)
    return f"{display_label}\nidentity {identity} | mapped C-alpha RMSD {aligned_entry.mapped_ca_rmsd:.2f} A"


def _style_for_entry(entry: PanelEntry) -> tuple[str, int, str]:
    color, transparency, default_label = _MODEL_STYLES.get(
        entry.selection_stratum, ("#009E73", 0, entry.selection_stratum)
    )
    return color, transparency, entry.display_label or default_label


def _overlay_legend_handles(entries: list[_AlignedOverlayEntry]) -> list[Patch]:
    handles = [Patch(facecolor=_REFERENCE_COLOR, alpha=0.58, label="ec86kit/7V9U reference")]
    seen_labels: set[str] = set()
    for aligned_entry in entries:
        color, _transparency, display_label = _style_for_entry(aligned_entry.entry)
        if display_label in seen_labels:
            continue
        seen_labels.add(display_label)
        handles.append(Patch(facecolor=color, alpha=0.92, label=display_label))
    return handles


def _identity_title_text(entry: PanelEntry) -> str:
    if entry.full_sequence_identity_percent is None:
        return "not available"
    return f"{entry.full_sequence_identity_percent:.1f}%"


def _format_optional_float(value: float | None) -> str:
    return "not_available" if value is None else f"{value:.3f}"


def _safe_model_name(candidate_id: str, selection_stratum: str) -> str:
    raw = f"{candidate_id}_{selection_stratum}"
    return "".join(character if character.isalnum() or character in {"_", "-"} else "_" for character in raw)


def _relative_chimerax_path(target_path: Path, *, script_path: Path) -> str:
    if not target_path.is_absolute():
        return str(target_path)
    return os.path.relpath(target_path, start=script_path.parent)


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
            timeout=180,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0
