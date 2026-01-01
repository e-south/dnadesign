"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/cruncher/workflows/parse_workflow.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from dnadesign.cruncher.config.schema_v2 import CruncherConfig
from dnadesign.cruncher.io.plots.pssm import plot_pwm
from dnadesign.cruncher.services.run_service import (
    update_run_index_from_manifest,
    update_run_index_from_status,
)
from dnadesign.cruncher.store.catalog_index import CatalogIndex
from dnadesign.cruncher.store.catalog_store import CatalogMotifStore
from dnadesign.cruncher.store.lockfile import read_lockfile, validate_lockfile, verify_lockfile_hashes
from dnadesign.cruncher.store.motif_store import MotifRef
from dnadesign.cruncher.utils.labels import build_run_name, regulator_sets
from dnadesign.cruncher.utils.manifest import build_run_manifest, write_manifest
from dnadesign.cruncher.utils.run_status import RunStatusWriter

logger = logging.getLogger(__name__)


def _store(cfg: CruncherConfig, config_path: Path):
    return CatalogMotifStore(
        config_path.parent / cfg.motif_store.catalog_root,
        pwm_source=cfg.motif_store.pwm_source,
        site_kinds=cfg.motif_store.site_kinds,
        combine_sites=cfg.motif_store.combine_sites,
        site_window_lengths=cfg.motif_store.site_window_lengths,
        site_window_center=cfg.motif_store.site_window_center,
        min_sites_for_pwm=cfg.motif_store.min_sites_for_pwm,
        allow_low_sites=cfg.motif_store.allow_low_sites,
    )


def _lockmap_for(cfg: CruncherConfig, config_path: Path) -> dict[str, object]:
    lock_root = config_path.parent / cfg.motif_store.catalog_root
    lock_path = lock_root / "locks" / f"{config_path.stem}.lock.json"
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
        catalog_root=lock_root,
        expected_pwm_source=cfg.motif_store.pwm_source,
    )
    return lockfile.resolved


def run_parse(cfg: CruncherConfig, config_path: Path) -> None:
    """
    Load each PWM for all regulators, generate a logo (PNG) and print summary
    (length, information bits, and full log-odds matrix) to stdout.

    Inputs:
      - cfg.regulator_sets       : list of TF groups; we flatten all TF names
      - cfg.parse.plot.bits_mode : "information" or "probability" (passed to plot_pwm)
      - cfg.parse.plot.dpi       : DPI for each saved logo

    Outputs (flat under `base_out`):
      - <base_out>/motif_logos/<tf>_logo.png  (one per TF)
      - Console: "[OK] <tf> L=<length> bits=<info_bits>"
                 followed by the log-odds matrix (pandas DataFrame) printout

    Error behavior:
      - If the motif store cannot resolve a PWM, we allow that exception to propagate.
    """
    store = _store(cfg, config_path)

    # Prepare output folder
    out_base = config_path.parent / Path(cfg.out_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    lockmap = _lockmap_for(cfg, config_path)
    catalog_root = config_path.parent / cfg.motif_store.catalog_root
    catalog = CatalogIndex.load(catalog_root)
    lock_root = catalog_root / "locks"
    lock_path = lock_root / f"{config_path.stem}.lock.json"

    for set_index, group in enumerate(regulator_sets(cfg.regulator_sets), start=1):
        if not group:
            raise ValueError(f"regulator_sets[{set_index}] is empty.")
        seen: set[str] = set()
        tfs = [tf for tf in group if not (tf in seen or seen.add(tf))]

        run_dir = out_base / build_run_name("parse", tfs, set_index=set_index)
        run_dir.mkdir(parents=True, exist_ok=True)
        status_writer = RunStatusWriter(
            path=run_dir / "run_status.json",
            stage="parse",
            run_dir=run_dir,
            payload={
                "config_path": str(config_path.resolve()),
                "status_message": "parsing",
                "regulator_set": {"index": set_index, "tfs": tfs},
            },
        )
        update_run_index_from_status(config_path, run_dir, status_writer.payload)

        artifacts: list[str] = []
        for tf in sorted(tfs):
            entry = lockmap.get(tf)
            if entry is None:
                raise ValueError(f"Missing lock entry for TF '{tf}'")
            ref = MotifRef(source=entry.source, motif_id=entry.motif_id)
            pwm = store.get_pwm(ref)

            plot_pwm(
                pwm,
                mode=cfg.parse.plot.bits_mode,
                out=run_dir / f"{tf}_logo.png",
                dpi=cfg.parse.plot.dpi,
            )
            artifacts.append(f"{tf}_logo.png")

            length = pwm.length
            bits = pwm.information_bits()
            logger.info("[OK] %s L=%d bits=%.1f", tf, length, bits)

            log_odds = pwm.log_odds()
            df_lo = pd.DataFrame(
                log_odds,
                columns=list(pwm.alphabet),
                index=[f"pos{i + 1}" for i in range(log_odds.shape[0])],
            )
            logger.debug("Log-odds matrix for %s:\n%s", tf, df_lo.to_string(float_format=lambda x: f"{x:.3f}"))

        set_lockmap = {tf: lockmap[tf] for tf in tfs}
        manifest = build_run_manifest(
            stage="parse",
            cfg=cfg,
            config_path=config_path,
            lock_path=lock_path,
            lockmap=set_lockmap,
            catalog=catalog,
            run_dir=run_dir,
            artifacts=artifacts,
            extra={
                "sequence_length": cfg.sample.init.length if cfg.sample else None,
                "regulator_set": {"index": set_index, "tfs": tfs},
            },
        )
        manifest_path = write_manifest(run_dir, manifest)
        update_run_index_from_manifest(config_path, run_dir, manifest)
        logger.info("Parse stage complete: %s", run_dir)
        logger.info("Wrote run manifest → %s", manifest_path.relative_to(run_dir.parent))
        status_writer.finish(status="completed", artifacts=artifacts)
        update_run_index_from_status(config_path, run_dir, status_writer.payload)
