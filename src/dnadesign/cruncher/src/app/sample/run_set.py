"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/sample/run_set.py

Public sampling entrypoint that delegates to run-set execution stages.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.cruncher.app.sample.artifacts import _save_config
from dnadesign.cruncher.app.sample.preflight import (
    ConfigError,
    RunError,
    _assert_init_length_fits_pwms,
    _assert_sequence_buffers_aligned,
    _assert_trace_meta_aligned,
    _core_def_hash,
    _mcmc_cooling_payload,
    _resolve_elite_pool_size,
    _resolve_final_softmin_beta,
    _softmin_schedule_payload,
    _validate_objective_preflight,
)
from dnadesign.cruncher.app.sample.resources import _load_pwms_for_set
from dnadesign.cruncher.app.sample.run_set_execution import run_sample_for_set
from dnadesign.cruncher.config.schema_v3 import CruncherConfig, SampleConfig

_REEXPORTED_PREFLIGHT = (
    ConfigError,
    RunError,
    _assert_init_length_fits_pwms,
    _assert_sequence_buffers_aligned,
    _assert_trace_meta_aligned,
    _core_def_hash,
    _mcmc_cooling_payload,
    _resolve_elite_pool_size,
    _resolve_final_softmin_beta,
    _softmin_schedule_payload,
    _validate_objective_preflight,
    _load_pwms_for_set,
    _save_config,
)


def _run_sample_for_set(
    cfg: CruncherConfig,
    config_path: Path,
    *,
    set_index: int,
    set_count: int,
    include_set_index: bool,
    tfs: list[str],
    lockmap: dict[str, object],
    sample_cfg: SampleConfig,
    stage: str = "sample",
    run_kind: str | None = None,
    force_overwrite: bool = False,
    progress_bar: bool = True,
    progress_every: int = 0,
    register_run_in_index: bool = True,
) -> Path:
    return run_sample_for_set(
        cfg=cfg,
        config_path=config_path,
        set_index=set_index,
        set_count=set_count,
        include_set_index=include_set_index,
        tfs=tfs,
        lockmap=lockmap,
        sample_cfg=sample_cfg,
        stage=stage,
        run_kind=run_kind,
        force_overwrite=force_overwrite,
        progress_bar=progress_bar,
        progress_every=progress_every,
        register_run_in_index=register_run_in_index,
        load_pwms_for_set=_load_pwms_for_set,
        save_config=_save_config,
    )
