"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/sample/resources.py

Resolve motif resources and lockfile data for sampling workflows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from pathlib import Path

from dnadesign.cruncher.config.schema_v3 import CruncherConfig, SampleConfig
from dnadesign.cruncher.core.pwm import PWM
from dnadesign.cruncher.core.pwm_window import select_pwm_window
from dnadesign.cruncher.store.catalog_store import CatalogMotifStore
from dnadesign.cruncher.store.lockfile import read_lockfile, validate_lockfile, verify_lockfile_hashes
from dnadesign.cruncher.store.motif_store import MotifRef
from dnadesign.cruncher.utils.paths import resolve_catalog_root, resolve_lock_path

logger = logging.getLogger(__name__)


def _store(cfg: CruncherConfig, config_path: Path):
    return CatalogMotifStore(
        resolve_catalog_root(config_path, cfg.catalog.catalog_root),
        pwm_source=cfg.catalog.pwm_source,
        site_kinds=cfg.catalog.site_kinds,
        combine_sites=cfg.catalog.combine_sites,
        site_window_lengths=cfg.catalog.site_window_lengths,
        site_window_center=cfg.catalog.site_window_center,
        apply_pwm_window=False,
        min_sites_for_pwm=cfg.catalog.min_sites_for_pwm,
        allow_low_sites=cfg.catalog.allow_low_sites,
        pseudocounts=cfg.catalog.pseudocounts,
    )


def _apply_sampling_pwm_window(*, tf_name: str, pwm: PWM, sample_cfg: SampleConfig) -> PWM:
    minw = sample_cfg.motif_width.minw
    maxw = sample_cfg.motif_width.maxw
    strategy = sample_cfg.motif_width.strategy

    if minw is not None and pwm.length < minw:
        raise ValueError(f"PWM for TF '{tf_name}' has length {pwm.length}, below sample.motif_width.minw={minw}.")

    max_allowed = sample_cfg.sequence_length if maxw is None else min(maxw, sample_cfg.sequence_length)
    minw_label = "none" if minw is None else str(minw)
    maxw_label = "none" if maxw is None else str(maxw)
    if pwm.length <= max_allowed:
        logger.info(
            "Sampling PWM width %s: source=%dbp effective=%dbp cap=%dbp minw=%s maxw=%s strategy=%s action=unchanged",
            tf_name,
            pwm.length,
            pwm.length,
            max_allowed,
            minw_label,
            maxw_label,
            strategy,
        )
        return pwm

    trimmed = select_pwm_window(pwm, length=max_allowed, strategy=strategy)
    logger.info(
        "Sampling PWM width %s: source=%dbp effective=%dbp cap=%dbp minw=%s maxw=%s strategy=%s action=trimmed",
        tf_name,
        pwm.length,
        trimmed.length,
        max_allowed,
        minw_label,
        maxw_label,
        strategy,
    )
    return trimmed


def _lockmap_for(cfg: CruncherConfig, config_path: Path) -> dict[str, object]:
    catalog_root = resolve_catalog_root(config_path, cfg.catalog.catalog_root)
    lock_path = resolve_lock_path(config_path)
    if not lock_path.exists():
        raise ValueError(f"Lockfile is required: {lock_path}. Run `cruncher lock {config_path.name}`.")
    lockfile = read_lockfile(lock_path)
    required = {tf for group in cfg.regulator_sets for tf in group}
    validate_lockfile(
        lockfile,
        expected_pwm_source=cfg.catalog.pwm_source,
        expected_site_kinds=cfg.catalog.site_kinds,
        expected_combine_sites=cfg.catalog.combine_sites,
        required_tfs=required,
    )
    verify_lockfile_hashes(
        lockfile=lockfile,
        catalog_root=catalog_root,
        expected_pwm_source=cfg.catalog.pwm_source,
    )
    return lockfile.resolved


def _load_pwms_for_set(
    *,
    cfg: CruncherConfig,
    config_path: Path,
    tfs: list[str],
    lockmap: dict[str, object],
) -> dict[str, PWM]:
    if cfg.sample is None:
        raise ValueError("sample config is required to load sampling PWMs.")
    store = _store(cfg, config_path)
    pwms: dict[str, PWM] = {}
    for tf in sorted(tfs):
        logger.debug("  Loading PWM for %s", tf)
        entry = lockmap.get(tf)
        if entry is None:
            raise ValueError(f"Missing lock entry for TF '{tf}'")
        ref = MotifRef(source=entry.source, motif_id=entry.motif_id)
        pwms[tf] = _apply_sampling_pwm_window(
            tf_name=tf,
            pwm=store.get_pwm(ref),
            sample_cfg=cfg.sample,
        )
    return pwms
