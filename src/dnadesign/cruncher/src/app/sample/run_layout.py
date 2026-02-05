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
from dnadesign.cruncher.artifacts.layout import (
    build_run_dir,
    ensure_run_dirs,
    manifest_path,
    run_group_label,
    status_path,
)
from dnadesign.cruncher.artifacts.manifest import build_run_manifest, write_manifest
from dnadesign.cruncher.artifacts.status import RunStatusWriter
from dnadesign.cruncher.config.schema_v2 import CruncherConfig, SampleConfig
from dnadesign.cruncher.core.pvalue import logodds_cache_info
from dnadesign.cruncher.store.catalog_index import CatalogIndex
from dnadesign.cruncher.utils.paths import resolve_catalog_root, resolve_lock_path


@dataclass(frozen=True)
class RunLayout:
    run_dir: Path
    run_group: str
    stage_label: str
    status_writer: RunStatusWriter


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
    auto_opt_meta: dict[str, object] | None,
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
    )
    ensure_run_dirs(run_dir, meta=True, artifacts=True, live=sample_cfg.output.live_metrics)
    run_group = run_group_label(tfs, set_index, include_set_index=include_set_index)
    stage_label = stage.upper().replace("_", "-")

    status_writer = RunStatusWriter(
        path=status_path(run_dir),
        stage=stage,
        run_dir=run_dir,
        metrics_path=run_dir / "live" / "metrics.jsonl" if sample_cfg.output.live_metrics else None,
        payload={
            "config_path": str(config_path.resolve()),
            "status_message": "initializing",
            "regulator_set": {"index": set_index, "tfs": tfs, "count": set_count},
            "run_group": run_group,
            "run_kind": run_kind,
            "auto_opt": auto_opt_meta,
        },
    )
    update_run_index_from_status(
        config_path,
        run_dir,
        status_writer.payload,
        catalog_root=cfg.motif_store.catalog_root,
    )
    status_writer.update(
        status_message="loading_pwms",
        draws=sample_cfg.budget.draws,
        tune=sample_cfg.budget.tune,
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
    auto_opt_meta: dict[str, object] | None,
    stage: str,
    run_dir: Path,
    lockmap: dict[str, object],
    artifacts: list[dict[str, object]],
    optimizer: object,
    optimizer_kind: str,
    combine_resolved: str,
    beta_softmin_final: float | None,
    status_writer: RunStatusWriter,
) -> Path:
    catalog_root = resolve_catalog_root(config_path, cfg.motif_store.catalog_root)
    catalog = CatalogIndex.load(catalog_root)
    lock_path = resolve_lock_path(config_path)
    lock_snapshot_path = manifest_path(run_dir).parent / "lockfile.json"
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

    manifest = build_run_manifest(
        stage=stage,
        cfg=cfg,
        config_path=config_path,
        lock_path=lock_snapshot_path,
        lockmap={tf: lockmap[tf] for tf in tfs},
        catalog=catalog,
        run_dir=run_dir,
        artifacts=artifacts,
        extra={
            "sequence_length": sample_cfg.init.length,
            "seed": sample_cfg.rng.seed,
            "seed_effective": sample_cfg.rng.seed + set_index - 1,
            "record_tune": sample_cfg.output.trace.include_tune,
            "save_trace": sample_cfg.output.trace.save,
            "tune": sample_cfg.budget.tune,
            "draws": sample_cfg.budget.draws,
            "early_stop": sample_cfg.early_stop.model_dump(),
            "top_k": sample_cfg.elites.k,
            "elites": sample_cfg.elites.model_dump(),
            "regulator_set": {"index": set_index, "tfs": tfs, "count": set_count},
            "run_group": run_group,
            "run_kind": run_kind,
            "auto_opt": auto_opt_meta,
            "objective": {
                "score_scale": sample_cfg.objective.score_scale,
                "combine": combine_resolved,
                "bidirectional": sample_cfg.objective.bidirectional,
                "softmin": sample_cfg.objective.softmin.model_dump(),
                "softmin_beta_final_resolved": beta_softmin_final,
            },
            "optimizer": {
                "kind": optimizer_kind,
                "pt": sample_cfg.optimizers.pt.model_dump(),
            },
            "optimizer_stats": optimizer.stats() if hasattr(optimizer, "stats") else {},
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
        catalog_root=cfg.motif_store.catalog_root,
    )
    status_writer.finish(status="completed", artifacts=artifacts)
    update_run_index_from_status(
        config_path,
        run_dir,
        status_writer.payload,
        catalog_root=cfg.motif_store.catalog_root,
    )
    return manifest_path_out
