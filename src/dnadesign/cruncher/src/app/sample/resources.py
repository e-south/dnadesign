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

from dnadesign.cruncher.config.schema_v2 import CruncherConfig
from dnadesign.cruncher.core.pwm import PWM
from dnadesign.cruncher.store.catalog_store import CatalogMotifStore
from dnadesign.cruncher.store.lockfile import read_lockfile, validate_lockfile, verify_lockfile_hashes
from dnadesign.cruncher.store.motif_store import MotifRef
from dnadesign.cruncher.utils.paths import resolve_catalog_root, resolve_lock_path

logger = logging.getLogger(__name__)


def _store(cfg: CruncherConfig, config_path: Path):
    return CatalogMotifStore(
        resolve_catalog_root(config_path, cfg.motif_store.catalog_root),
        pwm_source=cfg.motif_store.pwm_source,
        site_kinds=cfg.motif_store.site_kinds,
        combine_sites=cfg.motif_store.combine_sites,
        site_window_lengths=cfg.motif_store.site_window_lengths,
        site_window_center=cfg.motif_store.site_window_center,
        pwm_window_lengths=cfg.motif_store.pwm_window_lengths,
        pwm_window_strategy=cfg.motif_store.pwm_window_strategy,
        min_sites_for_pwm=cfg.motif_store.min_sites_for_pwm,
        allow_low_sites=cfg.motif_store.allow_low_sites,
        pseudocounts=cfg.motif_store.pseudocounts,
    )


def _lockmap_for(cfg: CruncherConfig, config_path: Path) -> dict[str, object]:
    catalog_root = resolve_catalog_root(config_path, cfg.motif_store.catalog_root)
    lock_path = resolve_lock_path(config_path)
    if not lock_path.exists():
        raise ValueError(f"Lockfile is required: {lock_path}. Run `cruncher lock {config_path.name}`.")
    lockfile = read_lockfile(lock_path)
    required = {tf for group in cfg.regulator_sets for tf in group}
    validate_lockfile(
        lockfile,
        expected_pwm_source=cfg.motif_store.pwm_source,
        expected_site_kinds=cfg.motif_store.site_kinds,
        expected_combine_sites=cfg.motif_store.combine_sites,
        required_tfs=required,
    )
    verify_lockfile_hashes(
        lockfile=lockfile,
        catalog_root=catalog_root,
        expected_pwm_source=cfg.motif_store.pwm_source,
    )
    return lockfile.resolved


def _load_pwms_for_set(
    *,
    cfg: CruncherConfig,
    config_path: Path,
    tfs: list[str],
    lockmap: dict[str, object],
) -> dict[str, PWM]:
    store = _store(cfg, config_path)
    pwms: dict[str, PWM] = {}
    for tf in sorted(tfs):
        logger.debug("  Loading PWM for %s", tf)
        entry = lockmap.get(tf)
        if entry is None:
            raise ValueError(f"Missing lock entry for TF '{tf}'")
        ref = MotifRef(source=entry.source, motif_id=entry.motif_id)
        pwms[tf] = store.get_pwm(ref)
    return pwms
