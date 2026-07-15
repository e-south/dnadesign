"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/communication_visuals/pipeline.py

Orchestrate communication-facing Eco1 review visuals from materialized evidence.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .catalog import (
    COMMUNICATION_RUNTIME_PATH_NAMES,
    MOVIE_TARGET_PROPOSAL_BACKBONES,
    MOVIE_TARGET_PROTECTED_EVIDENCE,
    MOVIE_TARGET_SELECTED_ELECTROSTATICS,
    validated_movie_targets,
)
from .constraint_map import write_design_space_map
from .proposal_backbone_cycle import write_proposal_backbone_cycle
from .selected_electrostatic_cycle import write_selected_electrostatic_cycle
from .selected_panel import write_selected_panel
from .structural_screen import write_structural_screen
from .structure_set import read_foldcheck_structure_set
from .structure_story import write_structure_story


def write_communication_visuals(
    *,
    panel_root: Path,
    mask_set_path: Path,
    conservation_profile_path: Path,
    policy_positions_path: Path,
    triage_table_path: Path,
    selection_panel_path: Path,
    foldcheck_full_structure_set_path: Path,
    reference_structure_path: Path,
    alignment_reference_backbone_path: Path,
    reference_structure_format: str,
    mask_residues: list[dict[str, Any]],
    render_movie_ids: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Materialize the additive communication lane without changing EDA artifacts."""

    required_paths = (
        mask_set_path,
        conservation_profile_path,
        policy_positions_path,
        triage_table_path,
        selection_panel_path,
        reference_structure_path,
        alignment_reference_backbone_path,
        foldcheck_full_structure_set_path,
    )
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)
    panel_root.mkdir(parents=True, exist_ok=True)
    conservation_rows = _read_parquet_rows(conservation_profile_path)
    policy_position_rows = _read_parquet_rows(policy_positions_path)
    triage_rows = _read_parquet_rows(triage_table_path)
    selected_rows = _read_parquet_rows(selection_panel_path)
    movie_targets = validated_movie_targets(render_movie_ids)
    structure_set = read_foldcheck_structure_set(foldcheck_full_structure_set_path)
    rows = [
        write_design_space_map(
            panel_root=panel_root,
            conservation_rows=conservation_rows,
            policy_position_rows=policy_position_rows,
            mask_residues=mask_residues,
            mask_set_path=mask_set_path,
            conservation_profile_path=conservation_profile_path,
            policy_positions_path=policy_positions_path,
        )
    ]
    rows.extend(
        write_structure_story(
            panel_root=panel_root,
            reference_structure_path=reference_structure_path,
            reference_structure_format=reference_structure_format,
            mask_residues=mask_residues,
            policy_position_rows=policy_position_rows,
            mask_set_path=mask_set_path,
            policy_positions_path=policy_positions_path,
            render_requested=MOVIE_TARGET_PROTECTED_EVIDENCE in movie_targets,
        )
    )
    rows.extend(
        write_proposal_backbone_cycle(
            panel_root=panel_root,
            triage_rows=triage_rows,
            triage_table_path=triage_table_path,
            structure_set=structure_set,
            foldcheck_full_structure_set_path=foldcheck_full_structure_set_path,
            reference_backbone_path=alignment_reference_backbone_path,
            render_requested=MOVIE_TARGET_PROPOSAL_BACKBONES in movie_targets,
        )
    )
    rows.append(
        write_structural_screen(
            panel_root=panel_root,
            triage_rows=triage_rows,
            selected_rows=selected_rows,
            triage_table_path=triage_table_path,
            selection_panel_path=selection_panel_path,
        )
    )
    rows.append(
        write_selected_panel(
            panel_root=panel_root,
            triage_rows=triage_rows,
            selected_rows=selected_rows,
            policy_position_rows=policy_position_rows,
            triage_table_path=triage_table_path,
            selection_panel_path=selection_panel_path,
            policy_positions_path=policy_positions_path,
        )
    )
    rows.extend(
        write_selected_electrostatic_cycle(
            panel_root=panel_root,
            selected_rows=selected_rows,
            selection_panel_path=selection_panel_path,
            structure_set=structure_set,
            foldcheck_full_structure_set_path=foldcheck_full_structure_set_path,
            reference_structure_path=reference_structure_path,
            render_requested=MOVIE_TARGET_SELECTED_ELECTROSTATICS in movie_targets,
        )
    )
    _remove_unmanifested_visual_outputs(panel_root=panel_root, rows=rows)
    return rows


def _read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in pq.read_table(path).to_pylist()]


def _remove_unmanifested_visual_outputs(*, panel_root: Path, rows: list[dict[str, Any]]) -> None:
    expected_names = {
        Path(str(row["path"])).name
        for row in rows
        if Path(str(row.get("path") or "")).suffix.lower() in {".cxc", ".mp4", ".png", ".svg"}
    }
    for suffix in (".cxc", ".mp4", ".png", ".svg"):
        for path in panel_root.glob(f"*{suffix}"):
            if path.name not in expected_names:
                path.unlink()
    for pattern in ("eco1_*_render_manifest.yaml", "eco1_*_chimerax.log", ".eco1_*_frames"):
        for path in panel_root.glob(pattern):
            if path.name in COMMUNICATION_RUNTIME_PATH_NAMES:
                continue
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
