"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/sample/run_layout.py

Run layout and manifest helpers for sampling workflows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from dnadesign.cruncher.app.run_service import update_run_index_from_manifest, update_run_index_from_status
from dnadesign.cruncher.artifacts.atomic_write import atomic_write_json
from dnadesign.cruncher.artifacts.entries import artifact_entry
from dnadesign.cruncher.artifacts.layout import (
    build_run_dir,
    ensure_run_dirs,
    live_metrics_path,
    lockfile_snapshot_path,
    run_group_label,
    run_optimize_dir,
    status_path,
)
from dnadesign.cruncher.artifacts.manifest import build_run_manifest, write_manifest
from dnadesign.cruncher.artifacts.status import RunStatusWriter
from dnadesign.cruncher.config.schema_v3 import CruncherConfig, SampleConfig
from dnadesign.cruncher.core.pvalue import logodds_cache_info
from dnadesign.cruncher.store.catalog_index import CatalogIndex
from dnadesign.cruncher.utils.paths import resolve_catalog_root, resolve_lock_path


@dataclass(frozen=True)
class RunLayout:
    run_dir: Path
    run_group: str
    stage_label: str
    status_writer: RunStatusWriter


def _materialize_optimizer_stats(
    run_dir: Path,
    optimizer_stats: object,
    *,
    stage: str,
) -> tuple[dict[str, object], list[dict[str, object]], str | None]:
    if optimizer_stats is None:
        return {}, [], None
    if not isinstance(optimizer_stats, dict):
        raise ValueError("optimizer.stats() must return a dictionary when present.")
    manifest_stats: dict[str, object] = dict(optimizer_stats)
    if "move_stats" not in manifest_stats:
        return manifest_stats, [], None
    move_stats = manifest_stats.pop("move_stats")
    if move_stats is None:
        return manifest_stats, [], None
    if not isinstance(move_stats, list):
        raise ValueError("optimizer.stats()['move_stats'] must be a list.")
    rel_path = Path("optimize") / "optimizer_move_stats.json"
    sidecar_path = run_optimize_dir(run_dir) / "optimizer_move_stats.json"
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(sidecar_path, {"move_stats": move_stats})
    manifest_stats["move_stats_path"] = str(rel_path)
    manifest_stats["move_stats_rows"] = len(move_stats)
    return (
        manifest_stats,
        [
            artifact_entry(
                sidecar_path,
                run_dir,
                kind="metadata",
                label="Optimizer move stats (JSON)",
                stage=stage,
            )
        ],
        str(rel_path),
    )


def prepare_run_layout(
    *,
    cfg: CruncherConfig,
    config_path: Path,
    sample_cfg: SampleConfig,
    tfs: list[str],
    set_index: int,
    set_count: int,
    include_set_index: bool,
    stage: str,
    run_kind: str | None,
    chain_count: int,
    optimizer_kind: str,
) -> RunLayout:
    base_out = config_path.parent / Path(cfg.out_dir)
    base_out.mkdir(parents=True, exist_ok=True)
    run_dir = build_run_dir(
        config_path=config_path,
        out_dir=cfg.out_dir,
        stage=stage,
        tfs=tfs,
        set_index=set_index,
        include_set_index=include_set_index,
        slot="latest",
    )
    previous_dir = build_run_dir(
        config_path=config_path,
        out_dir=cfg.out_dir,
        stage=stage,
        tfs=tfs,
        set_index=set_index,
        include_set_index=include_set_index,
        slot="previous",
    )
    if previous_dir.exists():
        shutil.rmtree(previous_dir)
    if run_dir.exists():
        shutil.move(str(run_dir), previous_dir)
    ensure_run_dirs(run_dir, meta=True, artifacts=True, live=sample_cfg.output.live_metrics)
    run_group = run_group_label(tfs, set_index, include_set_index=include_set_index)
    stage_label = stage.upper().replace("_", "-")

    status_writer = RunStatusWriter(
        path=status_path(run_dir),
        stage=stage,
        run_dir=run_dir,
        metrics_path=live_metrics_path(run_dir) if sample_cfg.output.live_metrics else None,
        payload={
            "config_path": str(config_path.resolve()),
            "status_message": "initializing",
            "regulator_set": {"index": set_index, "tfs": tfs, "count": set_count},
            "run_group": run_group,
            "run_kind": run_kind,
        },
    )
    update_run_index_from_status(
        config_path,
        run_dir,
        status_writer.payload,
        catalog_root=cfg.catalog.catalog_root,
    )
    adapt_sweeps = int(sample_cfg.budget.tune)
    draws = int(sample_cfg.budget.draws)
    total_sweeps = adapt_sweeps + draws
    status_writer.update(
        status_message="loading_pwms",
        total_sweeps=total_sweeps,
        adapt_sweeps=adapt_sweeps,
        draws=draws,
        chains=chain_count,
        optimizer=optimizer_kind,
    )
    return RunLayout(
        run_dir=run_dir,
        run_group=run_group,
        stage_label=stage_label,
        status_writer=status_writer,
    )


def write_run_manifest_and_update(
    *,
    cfg: CruncherConfig,
    config_path: Path,
    sample_cfg: SampleConfig,
    tfs: list[str],
    set_index: int,
    set_count: int,
    run_group: str,
    run_kind: str | None,
    stage: str,
    run_dir: Path,
    lockmap: dict[str, object],
    artifacts: list[dict[str, object]],
    optimizer: object,
    optimizer_kind: str,
    combine_resolved: str,
    beta_softmin_final: float | None,
    min_per_tf_norm: float | None,
    min_per_tf_norm_auto: dict[str, object] | None,
    status_writer: RunStatusWriter,
) -> Path:
    catalog_root = resolve_catalog_root(config_path, cfg.catalog.catalog_root)
    catalog = CatalogIndex.load(catalog_root)
    lock_path = resolve_lock_path(config_path)
    lock_snapshot_path = lockfile_snapshot_path(run_dir)
    lock_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(lock_path, lock_snapshot_path)
    except Exception as exc:  # pragma: no cover - filesystem dependent
        raise ValueError(f"Failed to snapshot lockfile for run: {lock_snapshot_path}") from exc

    cache_info = logodds_cache_info()
    pvalue_cache = None
    if cache_info is not None:
        pvalue_cache = {
            "hits": int(cache_info.hits),
            "misses": int(cache_info.misses),
            "maxsize": int(cache_info.maxsize) if cache_info.maxsize is not None else None,
            "currsize": int(cache_info.currsize),
        }

    optimizer_stats = optimizer.stats() if hasattr(optimizer, "stats") else {}
    manifest_optimizer_stats, optimizer_stat_artifacts, optimizer_stats_path = _materialize_optimizer_stats(
        run_dir,
        optimizer_stats,
        stage=stage,
    )
    manifest_artifacts = list(artifacts)
    manifest_artifacts.extend(optimizer_stat_artifacts)

    manifest = build_run_manifest(
        stage=stage,
        cfg=cfg,
        config_path=config_path,
        lock_path=lock_snapshot_path,
        lockmap={tf: lockmap[tf] for tf in tfs},
        catalog=catalog,
        run_dir=run_dir,
        artifacts=manifest_artifacts,
        extra={
            "sequence_length": sample_cfg.sequence_length,
            "seed": sample_cfg.seed,
            "seed_effective": sample_cfg.seed + set_index - 1,
            "record_tune": sample_cfg.output.include_tune_in_sequences,
            "save_trace": sample_cfg.output.save_trace,
            "total_sweeps": int(sample_cfg.budget.tune + sample_cfg.budget.draws),
            "adapt_sweeps": int(sample_cfg.budget.tune),
            "draws": int(sample_cfg.budget.draws),
            "top_k": sample_cfg.elites.k,
            "elites": sample_cfg.elites.model_dump(),
            "elites_min_per_tf_norm_resolved": min_per_tf_norm,
            "elites_min_per_tf_norm_auto": min_per_tf_norm_auto,
            "regulator_set": {"index": set_index, "tfs": tfs, "count": set_count},
            "run_group": run_group,
            "run_kind": run_kind,
            "objective": {
                "score_scale": sample_cfg.objective.score_scale,
                "combine": combine_resolved,
                "bidirectional": sample_cfg.objective.bidirectional,
                "softmin": sample_cfg.objective.softmin.model_dump(),
                "softmin_final_beta_used": beta_softmin_final,
            },
            "optimizer": {
                "kind": optimizer_kind,
            },
            "optimizer_stats": manifest_optimizer_stats,
            "optimizer_stats_path": optimizer_stats_path,
            "objective_schedule_summary": optimizer.objective_schedule_summary()
            if hasattr(optimizer, "objective_schedule_summary")
            else {},
            "pvalue_cache": pvalue_cache,
        },
    )
    manifest_path_out = write_manifest(run_dir, manifest)
    update_run_index_from_manifest(
        config_path,
        run_dir,
        manifest,
        catalog_root=cfg.catalog.catalog_root,
    )
    status_writer.finish(status="completed", artifacts=manifest_artifacts)
    update_run_index_from_status(
        config_path,
        run_dir,
        status_writer.payload,
        catalog_root=cfg.catalog.catalog_root,
    )
    return manifest_path_out
