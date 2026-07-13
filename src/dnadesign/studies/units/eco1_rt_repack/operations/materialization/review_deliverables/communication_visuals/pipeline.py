"""Orchestrate communication-facing Eco1 review visuals from materialized evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .candidate_cycle import write_candidate_cycle
from .constraint_map import write_design_space_map
from .selected_panel import write_selected_panel
from .structural_screen import write_structural_screen
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
    reference_structure_format: str,
    mask_residues: list[dict[str, Any]],
    render_chimerax: bool,
) -> list[dict[str, Any]]:
    """Materialize the additive communication lane without changing EDA artifacts."""

    required_paths = (
        mask_set_path,
        conservation_profile_path,
        policy_positions_path,
        triage_table_path,
        selection_panel_path,
        reference_structure_path,
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
            render_chimerax=render_chimerax,
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
        write_candidate_cycle(
            panel_root=panel_root,
            selected_rows=selected_rows,
            selection_panel_path=selection_panel_path,
            foldcheck_full_structure_set_path=foldcheck_full_structure_set_path,
            reference_structure_path=reference_structure_path,
            render_chimerax=render_chimerax,
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
