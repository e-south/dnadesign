"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/core/pipeline/run_finalization.py

Run output consolidation and manifest writing for the pipeline.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from ..artifacts.pool import _hash_file
from ..run_manifest import PlanManifest, RunManifest
from ..run_metrics import write_run_metrics
from ..run_paths import (
    display_path,
    inputs_manifest_path,
    run_manifest_path,
    run_outputs_root,
    run_tables_root,
)
from .library_artifacts import write_library_artifacts
from .outputs import _consolidate_parts

if TYPE_CHECKING:
    from ...config import DenseGenConfig, ResolvedPlanItem
    from ..artifacts.library import LibraryArtifact
    from .plan_pools import PlanPoolSpec

log = logging.getLogger(__name__)


def finalize_run_outputs(
    *,
    cfg: DenseGenConfig,
    run_root: Path,
    run_root_str: str,
    cfg_path: Path,
    config_sha: str,
    seed: int,
    seeds: dict[str, int],
    chosen_solver: str | None,
    solver_attempt_timeout_seconds: float | None,
    solver_threads: int | None,
    dense_arrays_version: str | None,
    dense_arrays_version_source: str,
    plan_stats: dict[tuple[str, str], dict[str, int]],
    plan_order: list[tuple[str, str]],
    plan_leaderboards: dict[tuple[str, str], dict],
    plan_pools: dict[str, PlanPoolSpec],
    plan_items: list[ResolvedPlanItem],
    inputs_manifest_entries: dict[str, dict],
    library_source: str | None,
    library_artifact: LibraryArtifact | None,
    library_build_rows: list[dict],
    library_member_rows: list[dict],
    composition_rows: list[dict],
) -> None:
    outputs_root = run_outputs_root(run_root)
    tables_root = run_tables_root(run_root)
    _consolidate_parts(tables_root, part_glob="attempts_part-*.parquet", final_name="attempts.parquet")
    _consolidate_parts(tables_root, part_glob="solutions_part-*.parquet", final_name="solutions.parquet")

    pool_manifest_hash = None
    pool_manifest_path = outputs_root / "pools" / "pool_manifest.json"
    if pool_manifest_path.exists():
        pool_manifest_hash = _hash_file(pool_manifest_path)
    elif library_artifact is not None and library_artifact.pool_manifest_hash:
        pool_manifest_hash = library_artifact.pool_manifest_hash

    write_library_artifacts(
        library_source=library_source,
        library_artifact=library_artifact,
        library_build_rows=library_build_rows,
        library_member_rows=library_member_rows,
        outputs_root=outputs_root,
        cfg_path=cfg_path,
        run_id=str(cfg.run.id),
        run_root=run_root,
        config_hash=config_sha,
        pool_manifest_hash=pool_manifest_hash,
    )

    if composition_rows:
        composition_path = tables_root / "composition.parquet"
        existing_rows: list[dict] = []
        if composition_path.exists():
            try:
                existing_rows = pd.read_parquet(composition_path).to_dict("records")
            except Exception as exc:
                raise RuntimeError(f"Failed to read existing composition.parquet at {composition_path}: {exc}") from exc
        existing_keys = {
            (str(row.get("solution_id") or ""), int(row.get("placement_index") or 0)) for row in existing_rows
        }
        new_rows = [
            row
            for row in composition_rows
            if (str(row.get("solution_id") or ""), int(row.get("placement_index") or 0)) not in existing_keys
        ]
        pd.DataFrame(existing_rows + new_rows).to_parquet(composition_path, index=False)

    try:
        write_run_metrics(cfg=cfg, run_root=run_root)
    except Exception as exc:
        raise RuntimeError(f"Failed to write run_metrics.parquet: {exc}") from exc

    plan_quota_by_name = {str(item.name): int(item.quota) for item in plan_items}
    manifest_items: list[PlanManifest] = []
    for key in plan_order:
        plan_name = str(key[1])
        if plan_name not in plan_quota_by_name:
            raise RuntimeError(f"Missing quota mapping for plan `{plan_name}` while writing run manifest.")
        manifest_items.append(
            PlanManifest(
                input_name=key[0],
                plan_name=plan_name,
                quota=int(plan_quota_by_name[plan_name]),
                generated=plan_stats[key]["generated"],
                duplicates_skipped=plan_stats[key]["duplicates_skipped"],
                failed_solutions=plan_stats[key]["failed_solutions"],
                total_resamples=plan_stats[key]["total_resamples"],
                libraries_built=plan_stats[key]["libraries_built"],
                stall_events=plan_stats[key]["stall_events"],
                failed_min_count_per_tf=plan_stats[key]["failed_min_count_per_tf"],
                failed_required_regulators=plan_stats[key]["failed_required_regulators"],
                failed_min_count_by_regulator=plan_stats[key]["failed_min_count_by_regulator"],
                failed_min_required_regulators=plan_stats[key]["failed_min_required_regulators"],
                duplicate_solutions=plan_stats[key]["duplicate_solutions"],
                leaderboard_latest=plan_leaderboards.get(key),
            )
        )
    manifest = RunManifest(
        run_id=cfg.run.id,
        created_at=datetime.now(timezone.utc).isoformat(),
        schema_version=str(cfg.schema_version),
        config_sha256=config_sha,
        run_root=run_root_str,
        random_seed=seed,
        seed_stage_a=seeds.get("stage_a"),
        seed_stage_b=seeds.get("stage_b"),
        seed_solver=seeds.get("solver"),
        solver_backend=chosen_solver,
        solver_strategy=str(cfg.solver.strategy),
        solver_attempt_timeout_seconds=solver_attempt_timeout_seconds,
        solver_threads=solver_threads,
        solver_strands=str(cfg.solver.strands),
        dense_arrays_version=dense_arrays_version,
        dense_arrays_version_source=dense_arrays_version_source,
        total_quota=int(sum(item.quota for item in manifest_items)),
        items=manifest_items,
    )
    manifest_path = run_manifest_path(run_root)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_json(manifest_path)

    if inputs_manifest_entries:
        manifest_inputs: list[dict] = []
        for item in plan_items:
            spec = plan_pools[item.name]
            entry = inputs_manifest_entries.get(spec.pool_name)
            if entry is not None:
                manifest_inputs.append(entry)
        payload = {
            "schema_version": str(cfg.schema_version),
            "run_id": cfg.run.id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config_sha256": config_sha,
            "inputs": manifest_inputs,
            "library_sampling": cfg.generation.sampling.model_dump(),
        }
        inputs_manifest = inputs_manifest_path(run_root)
        inputs_manifest.parent.mkdir(parents=True, exist_ok=True)
        inputs_manifest.write_text(json.dumps(payload, indent=2, sort_keys=True))
        log.info(
            "Inputs manifest written: %s",
            display_path(inputs_manifest, run_root, absolute=False),
        )
