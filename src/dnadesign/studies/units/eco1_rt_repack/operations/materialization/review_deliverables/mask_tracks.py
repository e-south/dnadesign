"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/mask_tracks.py

ChimeraX mask-context deliverables for Eco1 review.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import shutil
import subprocess
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
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.mask_rows import (
    read_mask_residues,
)

from .molecular_scene_contract import chimerax_reference_complex_style_commands, molecular_visual_contract
from .structure_browser_common import reference_residue_number_by_canonical

_CHIMERAX_COLOR_ORDER = (
    ("protected", "#d9d2c3", "baseline fixed residues (clade 9 p25 + 5 A)"),
    ("non_fixed", "#6aa84f", "ProteinMPNN-designable residues"),
    ("evolutionarily_conserved_clade9_25pct_plurality", "#4d78a8", "clade 9 p25 plurality"),
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


def write_mask_structure_context(
    *,
    panel_root: Path,
    mask_set_path: Path,
    reference_structure_path: Path,
    reference_structure_format: str,
    mask_residues: list[dict[str, Any]] | None = None,
    render_png: bool = True,
) -> list[dict[str, Any]]:
    """Write a ChimeraX script and optionally a PNG if ChimeraX is available."""

    residues = mask_residues if mask_residues is not None else read_mask_residues(mask_set_path)
    script_path = panel_root / "mask_structure_context.cxc"
    orientation_template_path = panel_root / "mask_structure_context_orientation_template.cxc"
    png_path = panel_root / "mask_structure_context.png"
    render_manifest_path = panel_root / "mask_structure_context_render_manifest.yaml"
    reference_number_by_canonical = reference_residue_number_by_canonical(
        residues,
        reference_structure_format=reference_structure_format,
    )
    _write_chimerax_script(
        script_path,
        residues=residues,
        reference_structure_path=reference_structure_path,
        reference_number_by_canonical=reference_number_by_canonical,
        png_path=png_path,
    )
    _write_orientation_template_script(
        orientation_template_path,
        residues=residues,
        reference_structure_path=reference_structure_path,
        reference_number_by_canonical=reference_number_by_canonical,
    )
    input_hashes = file_hashes({"mask_set": mask_set_path, "reference_structure": reference_structure_path})
    source_tables = ["mask_set.yaml", reference_structure_path.name]
    rows = [
        make_deliverable_row(
            deliverable_id="mask_structure_context_script",
            section=SECTION_CONSTRAINT_EVIDENCE,
            artifact_kind="chimerax_script",
            status="rendered",
            path=script_path,
            source_tables=source_tables,
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
            title="A ChimeraX recipe reproduces the active Ec86 mask-context view",
        ),
        make_deliverable_row(
            deliverable_id="mask_structure_context_orientation_template",
            section=SECTION_CONSTRAINT_EVIDENCE,
            artifact_kind="chimerax_script",
            status="rendered",
            path=orientation_template_path,
            source_tables=source_tables,
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
            title="A manual ChimeraX template preserves the Ec86 mask colors while tuning the camera",
        ),
    ]
    if not render_png:
        had_existing_render_artifact = png_path.exists() or render_manifest_path.exists()
        if _existing_render_is_current(
            render_manifest_path=render_manifest_path,
            script_path=script_path,
            reference_structure_path=reference_structure_path,
            png_path=png_path,
        ):
            status = "reused_existing_optional_render"
            skip_reason = "Reusing an existing ChimeraX PNG; rendering was disabled for this materialization run."
        else:
            png_path.unlink(missing_ok=True)
            render_manifest_path.unlink(missing_ok=True)
            if had_existing_render_artifact:
                status = "skipped_stale_optional_render_removed"
                skip_reason = (
                    "A stale ChimeraX PNG or render manifest did not match the current recipe and was removed."
                )
            else:
                status = "skipped_optional_render_disabled"
                skip_reason = "ChimeraX PNG rendering was disabled for this materialization run."
    else:
        executable = _find_chimerax()
        if executable:
            png_path.unlink(missing_ok=True)
            render_manifest_path.unlink(missing_ok=True)
            chimerax_completed = _run_chimerax(executable=executable, script_path=script_path)
            status = "rendered" if chimerax_completed and png_path.exists() else "skipped_runtime_failed"
            if status == "rendered":
                _write_render_manifest(
                    render_manifest_path=render_manifest_path,
                    script_path=script_path,
                    reference_structure_path=reference_structure_path,
                    png_path=png_path,
                )
            skip_reason = (
                "" if status == "rendered" else "ChimeraX was found, but the command did not write the expected PNG."
            )
        else:
            png_path.unlink(missing_ok=True)
            render_manifest_path.unlink(missing_ok=True)
            status = "skipped_missing_runtime"
            skip_reason = "ChimeraX executable was not found on PATH or at the standard macOS app path."
    rows.append(
        make_deliverable_row(
            deliverable_id="mask_structure_context_png",
            section=SECTION_CONSTRAINT_EVIDENCE,
            artifact_kind="png",
            status=status,
            path=png_path,
            source_tables=source_tables,
            input_hashes=input_hashes,
            alt_text="Optional ChimeraX render of the Eco1 RT mask structure context.",
            description="Rendered ChimeraX PNG showing the Ec86 reference protein colored by mask category.",
            interpretation_limit="This render is a mask-context view, not a fold-check result.",
            role="optional_heavy",
            skip_reason=skip_reason,
            title="The Ec86 reference structure shows the fixed residues protected by the active mask",
        )
    )
    return rows


def _write_chimerax_script(
    path: Path,
    *,
    residues: list[dict[str, Any]],
    reference_structure_path: Path,
    reference_number_by_canonical: dict[int, int],
    png_path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Eco1 RT repack mask context",
        "# mask_policy_id: active_eco1_rt_protected_residue_mask",
        f"# orientation_preset_id: {_CHIMERAX_ORIENTATION_PRESET_ID}",
        "# Paths are relative to this script directory.",
        *_chimerax_scene_setup_lines(reference_structure_path=reference_structure_path, script_path=path),
    ]
    lines.extend(
        _chimerax_mask_color_lines(
            residues,
            reference_number_by_canonical=reference_number_by_canonical,
        )
    )
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
    reference_structure_path: Path,
    reference_number_by_canonical: dict[int, int],
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
        "# mask_policy_id: active_eco1_rt_protected_residue_mask",
        f"# orientation_preset_id: {_CHIMERAX_ORIENTATION_PRESET_ID}",
        "# Paths are relative to this script directory.",
        *_chimerax_scene_setup_lines(reference_structure_path=reference_structure_path, script_path=path),
    ]
    lines.extend(
        _chimerax_mask_color_lines(
            residues,
            reference_number_by_canonical=reference_number_by_canonical,
        )
    )
    lines.extend(
        [
            *_CHIMERAX_VIEW_COMMANDS,
            '2dlabels text "Ec86 reference" xpos 0.035 ypos 0.90 size 24 color black bgColor none',
            "view name eco1_publication_v1",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _chimerax_scene_setup_lines(*, reference_structure_path: Path, script_path: Path) -> list[str]:
    return [
        "set bgColor white",
        "camera ortho",
        "lighting soft",
        "graphics silhouettes true",
        f"open {_relative_chimerax_path(reference_structure_path, script_path=script_path)}",
        *chimerax_reference_complex_style_commands(),
        "cartoon style width 1.4 thick 0.22",
    ]


def _chimerax_mask_color_lines(
    residues: list[dict[str, Any]],
    *,
    reference_number_by_canonical: dict[int, int],
) -> list[str]:
    lines: list[str] = []
    for field, color, label in _CHIMERAX_COLOR_ORDER:
        positions = [
            reference_number_by_canonical[int(row["canonical_position"])]
            for row in residues
            if bool(row.get(field)) and int(row["canonical_position"]) in reference_number_by_canonical
        ]
        selector = _selector_for_positions(positions)
        if selector:
            lines.append(f"# {label}")
            selection = f"#1/A:{selector}"
            lines.extend(
                (
                    f"color {selection} {color} target s",
                    f"show {selection} atoms",
                    f"style {selection} stick",
                    f"color {selection} {color} target a",
                )
            )
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


def _write_render_manifest(
    *,
    render_manifest_path: Path,
    script_path: Path,
    reference_structure_path: Path,
    png_path: Path,
) -> None:
    """Record the exact script and all-atom source behind a ChimeraX still."""

    payload = {
        "schema_id": "eco1_rt.mask_structure_context_render",
        "schema_version": 1,
        "status": "rendered",
        "script_hash": sha256(script_path),
        "reference_structure_hash": sha256(reference_structure_path),
        "output": {
            "path": png_path.name,
            "sha256": sha256(png_path),
        },
        "visual_contract": molecular_visual_contract(),
    }
    render_manifest_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )


def _existing_render_is_current(
    *,
    render_manifest_path: Path,
    script_path: Path,
    reference_structure_path: Path,
    png_path: Path,
) -> bool:
    """Return whether an optional still matches the current recipe and source."""

    if not all(path.exists() for path in (render_manifest_path, script_path, reference_structure_path, png_path)):
        return False
    try:
        payload = yaml.safe_load(render_manifest_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return False
    if not isinstance(payload, dict):
        return False
    output = payload.get("output")
    if not isinstance(output, dict):
        return False
    return (
        payload.get("schema_id") == "eco1_rt.mask_structure_context_render"
        and payload.get("script_hash") == sha256(script_path)
        and payload.get("reference_structure_hash") == sha256(reference_structure_path)
        and payload.get("visual_contract") == molecular_visual_contract()
        and output.get("path") == png_path.name
        and output.get("sha256") == sha256(png_path)
    )
