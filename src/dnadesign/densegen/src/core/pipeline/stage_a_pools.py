"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/core/pipeline/stage_a_pools.py

Stage-A pool preparation helpers (build/load, plan-scoped pools, and artifacts).

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from ...config import DenseGenConfig
from ...utils import logging_utils
from ..artifacts.candidates import build_candidate_artifact, find_candidate_files, prepare_candidates_dir
from ..artifacts.pool import PoolData, build_pool_artifact, load_pool_data, pool_status_by_input
from ..run_paths import display_path
from .outputs import _emit_event
from .plan_pools import PlanPoolSource, PlanPoolSpec, build_plan_pools

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class StageAPoolState:
    pool_dir: Path
    pool_manifest: Path
    pool_data: dict[str, PoolData]
    plan_pools: dict[str, PlanPoolSpec]
    plan_pool_sources: dict[str, PlanPoolSource]
    source_cache: dict[str, PoolData]


def _active_input_names(plan_items: list) -> set[str]:
    names: set[str] = set()
    for item in plan_items:
        include_inputs = list(getattr(item, "include_inputs", []) or [])
        for input_name in include_inputs:
            value = str(input_name).strip()
            if value:
                names.add(value)
    return names


def prepare_stage_a_pools(
    *,
    cfg: DenseGenConfig,
    cfg_path: Path,
    run_root: Path,
    outputs_root: Path,
    rng,
    build_stage_a: bool,
    candidate_logging: bool,
    candidates_dir: Path,
    plan_items: list,
    events_path: Path,
    run_id: str,
    deps,
) -> StageAPoolState:
    requested_progress_style = str(getattr(cfg.logging, "progress_style", "stream"))
    progress_style, _progress_reason = logging_utils.resolve_progress_style(
        requested_progress_style,
        stdout=sys.stdout,
    )
    logging_utils.set_progress_style(progress_style)
    logging_utils.set_progress_enabled(progress_style in {"stream", "screen"})

    pool_dir = outputs_root / "pools"
    pool_manifest = pool_dir / "pool_manifest.json"
    pool_data: dict[str, PoolData] | None = None
    active_inputs = _active_input_names(plan_items)
    if not active_inputs:
        raise RuntimeError("No active Stage-A inputs resolved from generation plan.")

    if build_stage_a:
        if candidate_logging:
            try:
                existed = prepare_candidates_dir(candidates_dir, overwrite=False)
            except Exception as exc:
                raise RuntimeError(f"Failed to prepare candidate artifacts directory: {exc}") from exc
            candidates_label = display_path(candidates_dir, run_root, absolute=False)
            if existed:
                log.info(
                    "Appending candidate artifacts under %s (use dense run --fresh to reset).",
                    candidates_label,
                )
            else:
                log.info("Candidate mining artifacts will be written to %s", candidates_label)
        try:
            _pool_artifact, pool_data = build_pool_artifact(
                cfg=cfg,
                cfg_path=cfg_path,
                deps=deps,
                rng=rng,
                outputs_root=outputs_root,
                out_dir=pool_dir,
                overwrite=True,
                selected_inputs=active_inputs,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to build Stage-A TFBS pools: {exc}") from exc
        try:
            _emit_event(
                events_path,
                event="POOL_BUILT",
                payload={
                    "inputs": [
                        {
                            "name": pool.name,
                            "input_type": pool.input_type,
                            "pool_mode": pool.pool_mode,
                            "rows": int(pool.df.shape[0]) if pool.df is not None else int(len(pool.sequences)),
                        }
                        for pool in pool_data.values()
                    ]
                },
            )
        except Exception:
            log.debug("Failed to emit POOL_BUILT event.", exc_info=True)

    if not pool_manifest.exists():
        raise RuntimeError(
            "Stage-A pools missing or stale. Run `uv run dense stage-a build-pool --fresh` to regenerate pools."
        )
    if not build_stage_a:
        statuses = pool_status_by_input(cfg, cfg_path, run_root)
        stale = [status for name, status in statuses.items() if name in active_inputs and status.state != "present"]
        if stale:
            labels = ", ".join(sorted({status.name for status in stale}))
            raise RuntimeError(
                "Stage-A pools missing or stale for: "
                f"{labels}. Run `uv run dense stage-a build-pool --fresh` to regenerate pools."
            )
        stale_unused = sorted(
            name for name, status in statuses.items() if name not in active_inputs and status.state != "present"
        )
        if stale_unused:
            log.info("Ignoring stale Stage-A pools for unused inputs: %s", ", ".join(stale_unused))
    try:
        _pool_artifact, pool_data = load_pool_data(pool_dir)
    except Exception as exc:
        raise RuntimeError(f"Failed to load existing Stage-A pool artifacts: {exc}") from exc
    pool_label = display_path(pool_dir, run_root, absolute=False)
    log.info("Using Stage-A pools from %s", pool_label)

    plan_pools = build_plan_pools(plan_items=plan_items, pool_data=pool_data)
    plan_pool_sources = {plan_name: PlanPoolSource(name=spec.pool_name) for plan_name, spec in plan_pools.items()}
    source_cache = {spec.pool_name: spec.pool for spec in plan_pools.values()}

    if candidate_logging and build_stage_a:
        candidate_files = find_candidate_files(candidates_dir)
        if candidate_files:
            try:
                build_candidate_artifact(
                    candidates_dir=candidates_dir,
                    cfg_path=cfg_path,
                    run_id=str(run_id),
                    run_root=run_root,
                    overwrite=True,
                )
            except Exception as exc:
                raise RuntimeError(f"Failed to write candidate artifacts: {exc}") from exc
        else:
            candidates_label = display_path(candidates_dir, run_root, absolute=False)
            log.warning(
                "Candidate logging enabled but no candidate records were written under %s. "
                "Check keep_all_candidates_debug and PWM inputs.",
                candidates_label,
            )

    return StageAPoolState(
        pool_dir=pool_dir,
        pool_manifest=pool_manifest,
        pool_data=pool_data,
        plan_pools=plan_pools,
        plan_pool_sources=plan_pool_sources,
        source_cache=source_cache,
    )
