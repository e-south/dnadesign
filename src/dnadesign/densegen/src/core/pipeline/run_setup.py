"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/core/pipeline/run_setup.py

Run setup helpers for plan tracking and display mapping.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ..motif_labels import input_motifs
from ..run_paths import display_path


def build_display_map_by_input(
    *,
    plan_items: Iterable[object],
    plan_pools: dict,
    inputs: Iterable[object],
    cfg_path: Path,
) -> dict[str, dict[str, str]]:
    inputs_by_name = {inp.name: inp for inp in inputs}
    motifs_by_input: dict[str, list[tuple[str | None, str | None]]] = {}
    display_map_by_input: dict[str, dict[str, str]] = {}
    for item in plan_items:
        spec = plan_pools[item.name]
        mapping: dict[str, str] = {}
        for input_name in spec.include_inputs:
            inp = inputs_by_name.get(input_name)
            if inp is None:
                continue
            motifs = motifs_by_input.get(input_name)
            if motifs is None:
                motifs = list(input_motifs(inp, cfg_path))
                motifs_by_input[input_name] = motifs
            for motif_id, name in motifs:
                if motif_id and name and motif_id not in mapping:
                    mapping[motif_id] = name
        if mapping:
            display_map_by_input[spec.pool_name] = mapping
    return display_map_by_input


def init_plan_stats(
    *,
    plan_items: Iterable[object],
    plan_pools: dict,
    existing_counts: dict[tuple[str, str], int] | None,
) -> tuple[dict[tuple[str, str], dict[str, int]], list[tuple[str, str]]]:
    plan_stats: dict[tuple[str, str], dict[str, int]] = {}
    plan_order: list[tuple[str, str]] = []
    counts = existing_counts or {}
    for item in plan_items:
        spec = plan_pools[item.name]
        key = (spec.pool_name, item.name)
        plan_stats[key] = {
            "generated": int(counts.get(key, 0)),
            "duplicates_skipped": 0,
            "failed_solutions": 0,
            "total_resamples": 0,
            "libraries_built": 0,
            "stall_events": 0,
            "failed_min_count_per_tf": 0,
            "failed_required_regulators": 0,
            "failed_min_count_by_regulator": 0,
            "failed_min_required_regulators": 0,
            "duplicate_solutions": 0,
        }
        plan_order.append(key)
    return plan_stats, plan_order


def init_state_counts(
    *,
    plan_items: Iterable[object],
    plan_pools: dict,
    existing_counts: dict[tuple[str, str], int] | None,
) -> dict[tuple[str, str], int]:
    state_counts: dict[tuple[str, str], int] = {}
    counts = existing_counts or {}
    for item in plan_items:
        spec = plan_pools[item.name]
        state_counts[(spec.pool_name, item.name)] = int(counts.get((spec.pool_name, item.name), 0))
    return state_counts


def validate_resume_outputs(
    *,
    resume: bool,
    existing_outputs: bool,
    outputs_root: Path,
    run_root: Path,
) -> None:
    if resume:
        if not existing_outputs:
            outputs_label = display_path(outputs_root, run_root, absolute=False)
            raise RuntimeError(
                f"resume=True requested but no outputs were found under {outputs_label}. "
                "Start a fresh run or remove resume=True."
            )
    else:
        if existing_outputs:
            outputs_label = display_path(outputs_root, run_root, absolute=False)
            raise RuntimeError(
                f"Existing outputs found under {outputs_label}. Explicit resume is required to continue this run."
            )
