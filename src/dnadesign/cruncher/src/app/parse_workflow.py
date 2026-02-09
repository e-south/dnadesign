"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/parse_workflow.py

Parse locked motifs into normalized catalog artifacts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from dnadesign.cruncher.app.parse_signature import compute_parse_signature
from dnadesign.cruncher.artifacts.atomic_write import atomic_write_json
from dnadesign.cruncher.artifacts.layout import (
    build_run_dir,
    ensure_run_dirs,
    lockfile_snapshot_path,
    parse_manifest_path,
    pwm_summary_path,
    run_group_label,
)
from dnadesign.cruncher.artifacts.manifest import build_run_manifest
from dnadesign.cruncher.config.schema_v3 import CruncherConfig
from dnadesign.cruncher.core.labels import regulator_sets
from dnadesign.cruncher.store.catalog_index import CatalogIndex
from dnadesign.cruncher.store.catalog_store import CatalogMotifStore
from dnadesign.cruncher.store.lockfile import (
    Lockfile,
    read_lockfile,
    validate_lockfile,
    verify_lockfile_hashes,
)
from dnadesign.cruncher.store.motif_store import MotifRef
from dnadesign.cruncher.utils.paths import resolve_catalog_root, resolve_lock_path
from dnadesign.cruncher.viz.logos import pwm_provenance_summary

logger = logging.getLogger(__name__)


def _store(cfg: CruncherConfig, config_path: Path):
    return CatalogMotifStore(
        resolve_catalog_root(config_path, cfg.catalog.catalog_root),
        pwm_source=cfg.catalog.pwm_source,
        site_kinds=cfg.catalog.site_kinds,
        combine_sites=cfg.catalog.combine_sites,
        site_window_lengths=cfg.catalog.site_window_lengths,
        site_window_center=cfg.catalog.site_window_center,
        pwm_window_lengths=cfg.catalog.pwm_window_lengths,
        pwm_window_strategy=cfg.catalog.pwm_window_strategy,
        min_sites_for_pwm=cfg.catalog.min_sites_for_pwm,
        allow_low_sites=cfg.catalog.allow_low_sites,
        pseudocounts=cfg.catalog.pseudocounts,
    )


def _lockmap_for(cfg: CruncherConfig, config_path: Path) -> tuple[Lockfile, dict[str, object]]:
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
    return lockfile, lockfile.resolved


def run_parse(
    cfg: CruncherConfig,
    config_path: Path,
    *,
    force_overwrite: bool = False,
) -> None:
    """
    Load each PWM for all regulators and validate cached motifs, logging a summary
    (length, information bits, and PWM window metadata) to stdout.

    Inputs:
      - cfg.regulator_sets       : list of TF groups; we flatten all TF names
    Outputs (per run):
      - Console: "[OK] <tf> L=<length> bits=<info_bits>"
                 followed by optional log-odds matrices in debug logs.

    Error behavior:
      - If the motif store cannot resolve a PWM, we allow that exception to propagate.
    """
    catalog_root = resolve_catalog_root(config_path, cfg.catalog.catalog_root)

    store = _store(cfg, config_path)

    groups = regulator_sets(cfg.regulator_sets)
    if not groups:
        raise ValueError("parse requires at least one regulator set.")
    set_count = len(groups)
    include_set_index = set_count > 1

    lockfile, lockmap = _lockmap_for(cfg, config_path)
    catalog = CatalogIndex.load(catalog_root)
    lock_path = resolve_lock_path(config_path)
    logger.info("Using lockfile: %s", lock_path)
    logger.info(
        "PWM config: source=%s combine_sites=%s site_kinds=%s",
        cfg.catalog.pwm_source,
        cfg.catalog.combine_sites,
        cfg.catalog.site_kinds,
    )

    for set_index, group in enumerate(groups, start=1):
        if not group:
            raise ValueError(f"regulator_sets[{set_index}] is empty.")
        seen: set[str] = set()
        tfs = [tf for tf in group if not (tf in seen or seen.add(tf))]
        signature, signature_payload = compute_parse_signature(
            cfg=cfg,
            lockfile=lockfile,
            tfs=tfs,
        )

        run_group = run_group_label(tfs, set_index, include_set_index=include_set_index)
        run_dir = build_run_dir(
            config_path=config_path,
            out_dir=cfg.out_dir,
            stage="parse",
            tfs=tfs,
            set_index=set_index,
            include_set_index=include_set_index,
        )
        if run_dir.exists():
            if not run_dir.is_dir():
                raise ValueError(f"Parse output path exists and is not a directory: {run_dir}")
            has_entries = any(run_dir.iterdir())
            if has_entries:
                if not force_overwrite:
                    raise ValueError(
                        f"Parse output directory already exists: {run_dir}. "
                        "Re-run with --force-overwrite to replace it."
                    )
                shutil.rmtree(run_dir)
        ensure_run_dirs(run_dir, meta=True)
        pwm_rows: list[dict[str, object]] = []

        for tf in sorted(tfs):
            locked = lockmap.get(tf)
            if locked is None:
                raise ValueError(f"Missing lock entry for TF '{tf}'")
            ref = MotifRef(source=locked.source, motif_id=locked.motif_id)
            catalog_entry = catalog.entries.get(f"{ref.source}:{ref.motif_id}")
            if catalog_entry is None:
                raise ValueError(f"Catalog entry missing for {ref.source}:{ref.motif_id}")
            provenance = pwm_provenance_summary(
                pwm_source=cfg.catalog.pwm_source,
                entry=catalog_entry,
                catalog=catalog,
                combine_sites=cfg.catalog.combine_sites,
                site_kinds=cfg.catalog.site_kinds,
            )
            logger.info("PWM %s ← %s:%s | %s", tf, ref.source, ref.motif_id, provenance)

            pwm = store.get_pwm(ref)

            length = pwm.length
            bits = pwm.information_bits()
            if pwm.nsites is not None:
                logger.info("[OK] %s L=%d bits=%.1f n=%d", tf, length, bits, pwm.nsites)
            else:
                logger.info("[OK] %s L=%d bits=%.1f", tf, length, bits)
            if pwm.source_length is not None and pwm.window_start is not None:
                logger.info(
                    "PWM window %s: %d:%d/%d (%s)",
                    tf,
                    pwm.window_start,
                    pwm.window_start + pwm.length,
                    pwm.source_length,
                    pwm.window_strategy or "max_info",
                )
            pwm_rows.append(
                {
                    "tf": tf,
                    "source": ref.source,
                    "motif_id": ref.motif_id,
                    "length": int(length),
                    "information_bits": float(bits),
                    "nsites": int(pwm.nsites) if pwm.nsites is not None else None,
                    "window_start": int(pwm.window_start) if pwm.window_start is not None else None,
                    "source_length": int(pwm.source_length) if pwm.source_length is not None else None,
                    "window_strategy": pwm.window_strategy,
                }
            )

            log_odds = pwm.log_odds()
            df_lo = pd.DataFrame(
                log_odds,
                columns=list(pwm.alphabet),
                index=[f"pos{i + 1}" for i in range(log_odds.shape[0])],
            )
            logger.debug(
                "Log-odds matrix for %s:\n%s",
                tf,
                df_lo.to_string(float_format=lambda x: f"{x:.3f}"),
            )

        set_lockmap = {tf: lockmap[tf] for tf in tfs}
        manifest = build_run_manifest(
            stage="parse",
            cfg=cfg,
            config_path=config_path,
            lock_path=lock_path,
            lockmap=set_lockmap,
            catalog=catalog,
            run_dir=run_dir,
            artifacts=[],
            extra={
                "sequence_length": cfg.sample.sequence_length if cfg.sample else None,
                "regulator_set": {"index": set_index, "tfs": tfs, "count": set_count},
                "run_group": run_group,
                "parse_signature": signature,
                "parse_inputs": signature_payload,
            },
        )
        lock_snapshot = lockfile_snapshot_path(run_dir)
        lock_snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(lock_path, lock_snapshot)

        parse_manifest_file = parse_manifest_path(run_dir)
        atomic_write_json(parse_manifest_file, manifest)

        pwm_summary_file = pwm_summary_path(run_dir)
        atomic_write_json(
            pwm_summary_file,
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "regulator_set": {"index": set_index, "tfs": tfs, "count": set_count},
                "run_group": run_group,
                "rows": pwm_rows,
            },
        )
        logger.info("Parse stage complete: %s", run_dir)
        logger.info("Wrote parse manifest → %s", parse_manifest_file.relative_to(run_dir))
