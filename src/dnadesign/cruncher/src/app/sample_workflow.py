"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/sample_workflow.py

Coordinate sampling runs across regulator sets.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import signal
from contextlib import contextmanager
from pathlib import Path

from dnadesign.cruncher.app.sample.artifacts import _format_run_path
from dnadesign.cruncher.app.sample.resources import _load_pwms_for_set, _lockmap_for, _store
from dnadesign.cruncher.app.sample.run_set import _resolve_final_softmin_beta, _run_sample_for_set
from dnadesign.cruncher.app.target_service import (
    has_blocking_target_errors,
    target_statuses,
)
from dnadesign.cruncher.artifacts.layout import config_used_path
from dnadesign.cruncher.config.schema_v3 import CruncherConfig
from dnadesign.cruncher.core.labels import regulator_sets
from dnadesign.cruncher.utils.arviz_cache import ensure_arviz_data_dir
from dnadesign.cruncher.utils.paths import resolve_catalog_root
from dnadesign.cruncher.viz.mpl import ensure_mpl_cache

logger = logging.getLogger(__name__)

__all__ = [
    "_format_run_path",
    "_load_pwms_for_set",
    "_lockmap_for",
    "_resolve_final_softmin_beta",
    "_run_sample_for_set",
    "_sigterm_as_keyboard_interrupt",
    "_store",
    "run_sample",
]


@contextmanager
def _sigterm_as_keyboard_interrupt():
    sigterm = signal.SIGTERM
    previous = signal.getsignal(sigterm)

    def _handler(signum: int, frame: object) -> None:
        raise KeyboardInterrupt("SIGTERM")

    signal.signal(sigterm, _handler)
    try:
        yield
    finally:
        signal.signal(sigterm, previous)


def run_sample(
    cfg: CruncherConfig,
    config_path: Path,
) -> None:
    """
    Run MCMC sampler, save config/meta plus artifacts (trace.nc, sequences.parquet, elites.*).
    Each regulator set is sampled independently for clear provenance.
    """
    if cfg.sample is None:
        raise ValueError("sample section is required for sample")
    with _sigterm_as_keyboard_interrupt():
        if cfg.sample.output.save_trace:
            catalog_root = resolve_catalog_root(config_path, cfg.catalog.catalog_root)
            ensure_mpl_cache(catalog_root)
            ensure_arviz_data_dir(catalog_root)
        lockmap = _lockmap_for(cfg, config_path)
        statuses = target_statuses(cfg=cfg, config_path=config_path)
        sample_cfg = cfg.sample
        groups = regulator_sets(cfg.regulator_sets)
        if not groups:
            raise ValueError("sample requires at least one regulator set.")
        set_count = len(groups)
        include_set_index = set_count > 1
        for set_index, group in enumerate(groups, start=1):
            if not group:
                raise ValueError(f"regulator_sets[{set_index}] is empty.")
            seen: set[str] = set()
            tfs = [tf for tf in group if not (tf in seen or seen.add(tf))]
            subset = [status for status in statuses if status.set_index == set_index]
            if has_blocking_target_errors(subset):
                details = "; ".join(f"{s.tf_name}:{s.status}" for s in subset if s.status not in {"ready", "warning"})
                raise ValueError(
                    f"Target readiness failed for set {set_index} ({details}). "
                    f"Run `cruncher targets status {config_path.name}` for details."
                )
            run_dir = _run_sample_for_set(
                cfg,
                config_path,
                set_index=set_index,
                set_count=set_count,
                include_set_index=include_set_index,
                tfs=tfs,
                lockmap=lockmap,
                sample_cfg=sample_cfg,
            )
            logger.info(
                "Sample outputs -> %s",
                _format_run_path(run_dir, base=config_path.parent),
            )
            logger.info(
                "Config used -> %s",
                _format_run_path(config_used_path(run_dir), base=config_path.parent),
            )
