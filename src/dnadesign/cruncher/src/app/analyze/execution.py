"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/analyze/execution.py

Builds run-scoped analysis execution context (metadata and artifact handles).

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from dnadesign.cruncher.analysis.layout import analysis_root, analysis_state_root, analysis_used_path
from dnadesign.cruncher.app.analyze.metadata import (
    SampleMeta,
    _analysis_id,
    _resolve_sample_meta,
    load_pwms_from_config,
)
from dnadesign.cruncher.app.analyze.optimizer_stats import _resolve_optimizer_stats
from dnadesign.cruncher.app.analyze.run_resolution import _resolve_run_dir
from dnadesign.cruncher.app.analyze.staging import analyze_lock_meta_path, recoverable_analyze_lock_reason
from dnadesign.cruncher.app.analyze_support import _load_run_artifacts_for_analysis, _resolve_tf_names
from dnadesign.cruncher.artifacts.atomic_write import atomic_write_json
from dnadesign.cruncher.artifacts.manifest import load_manifest
from dnadesign.cruncher.config.schema_v3 import CruncherConfig
from dnadesign.cruncher.utils.hashing import sha256_path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnalysisRunExecutionContext:
    run_name: str
    run_dir: Path
    manifest: dict[str, object]
    optimizer_stats: dict[str, object] | None
    pwms: dict[str, Any]
    used_cfg: dict[str, object]
    tf_names: list[str]
    sample_meta: SampleMeta
    analysis_id: str
    created_at: str
    analysis_root_path: Path
    tmp_root: Path
    analysis_used_file: Path
    require_random_baseline: bool
    sequences_df: pd.DataFrame
    elites_df: pd.DataFrame
    hits_df: pd.DataFrame
    baseline_df: pd.DataFrame
    baseline_hits_df: pd.DataFrame
    trace_idata: object | None
    elites_meta: dict[str, object]


def _verify_manifest_lockfile(manifest: dict[str, object]) -> None:
    lockfile_path = manifest.get("lockfile_path")
    lockfile_sha = manifest.get("lockfile_sha256")
    if not lockfile_path or not lockfile_sha:
        return
    lock_path = Path(str(lockfile_path))
    if not lock_path.exists():
        raise FileNotFoundError(f"Lockfile referenced by run manifest missing: {lock_path}")
    current_sha = sha256_path(lock_path)
    if str(current_sha) != str(lockfile_sha):
        raise ValueError("Lockfile checksum mismatch (run manifest does not match current lockfile).")


def _create_analyze_tmp_root(
    *,
    analysis_root_path: Path,
    run_name: str,
    analysis_id: str,
    created_at: str,
) -> Path:
    tmp_root = analysis_state_root(analysis_root_path) / "tmp"
    if tmp_root.exists():
        recoverable_reason = recoverable_analyze_lock_reason(tmp_root)
        if recoverable_reason is not None:
            logger.warning(
                "Recovering stale analyze lock for run '%s' at %s (%s).",
                run_name,
                tmp_root,
                recoverable_reason,
            )
            shutil.rmtree(tmp_root, ignore_errors=True)
        else:
            raise RuntimeError(
                f"Analyze already in progress for run '{run_name}' (lock: {tmp_root}). "
                "If no analyze is running, remove the stale analysis temp directory."
            )
    try:
        tmp_root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise RuntimeError(
            f"Analyze already in progress for run '{run_name}' (lock: {tmp_root}). "
            "If no analyze is running, remove the stale analysis temp directory."
        ) from exc
    atomic_write_json(
        analyze_lock_meta_path(tmp_root),
        {
            "analysis_id": analysis_id,
            "run": run_name,
            "created_at": created_at,
            "pid": os.getpid(),
        },
    )
    return tmp_root


def resolve_analysis_run_execution_context(
    *,
    cfg: CruncherConfig,
    config_path: Path,
    run_name: str,
) -> AnalysisRunExecutionContext:
    run_dir = _resolve_run_dir(cfg, config_path, run_name)
    manifest = load_manifest(run_dir)
    optimizer_stats_raw = _resolve_optimizer_stats(manifest, run_dir)
    optimizer_stats = optimizer_stats_raw if isinstance(optimizer_stats_raw, dict) else None
    _verify_manifest_lockfile(manifest)
    pwms, used_cfg = load_pwms_from_config(run_dir)
    tf_names = _resolve_tf_names(used_cfg, pwms)
    sample_meta = _resolve_sample_meta(used_cfg, manifest)
    analysis_id = _analysis_id()
    created_at = datetime.now(timezone.utc).isoformat()

    analysis_root_path = analysis_root(run_dir)
    tmp_root = _create_analyze_tmp_root(
        analysis_root_path=analysis_root_path,
        run_name=run_name,
        analysis_id=analysis_id,
        created_at=created_at,
    )
    analysis_used_file = analysis_used_path(tmp_root)

    require_random_baseline = bool(cfg.sample is not None and cfg.sample.output.save_random_baseline)
    artifacts = _load_run_artifacts_for_analysis(
        run_dir,
        require_random_baseline=require_random_baseline,
    )

    return AnalysisRunExecutionContext(
        run_name=run_name,
        run_dir=run_dir,
        manifest=manifest,
        optimizer_stats=optimizer_stats,
        pwms=pwms,
        used_cfg=used_cfg,
        tf_names=tf_names,
        sample_meta=sample_meta,
        analysis_id=analysis_id,
        created_at=created_at,
        analysis_root_path=analysis_root_path,
        tmp_root=tmp_root,
        analysis_used_file=analysis_used_file,
        require_random_baseline=require_random_baseline,
        sequences_df=artifacts.sequences_df,
        elites_df=artifacts.elites_df,
        hits_df=artifacts.hits_df,
        baseline_df=artifacts.baseline_df,
        baseline_hits_df=artifacts.baseline_hits_df,
        trace_idata=artifacts.trace_idata,
        elites_meta=artifacts.elites_meta,
    )


__all__ = ["AnalysisRunExecutionContext", "resolve_analysis_run_execution_context"]
