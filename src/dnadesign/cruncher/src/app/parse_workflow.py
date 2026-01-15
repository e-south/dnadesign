"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/parse_workflow.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from dnadesign.cruncher.app.run_service import (
    update_run_index_from_manifest,
    update_run_index_from_status,
)
from dnadesign.cruncher.artifacts.layout import (
    build_run_dir,
    ensure_run_dirs,
    out_root,
    run_group_dir,
    run_group_label,
    status_path,
)
from dnadesign.cruncher.artifacts.manifest import build_run_manifest, load_manifest, write_manifest
from dnadesign.cruncher.artifacts.status import RunStatusWriter
from dnadesign.cruncher.config.schema_v2 import CruncherConfig
from dnadesign.cruncher.core.labels import regulator_sets
from dnadesign.cruncher.store.catalog_index import CatalogIndex
from dnadesign.cruncher.store.catalog_store import CatalogMotifStore
from dnadesign.cruncher.store.lockfile import (
    Lockfile,
    lockfile_fingerprint,
    read_lockfile,
    validate_lockfile,
    verify_lockfile_hashes,
)
from dnadesign.cruncher.store.motif_store import MotifRef
from dnadesign.cruncher.utils.hashing import sha256_bytes
from dnadesign.cruncher.utils.paths import resolve_catalog_root, resolve_lock_path
from dnadesign.cruncher.viz.logos import pwm_provenance_summary

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


def _lockmap_for(cfg: CruncherConfig, config_path: Path) -> tuple[Lockfile, dict[str, object]]:
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
    return lockfile, lockfile.resolved


def _parse_signature(
    *,
    cfg: CruncherConfig,
    lockfile: Lockfile,
    tfs: list[str],
) -> tuple[str, dict[str, object]]:
    lock_sig, _ = lockfile_fingerprint(lockfile)
    payload = {
        "tfs": sorted(tfs),
        "lockfile_fingerprint": lock_sig,
        "motif_store": {
            "pwm_source": cfg.motif_store.pwm_source,
            "combine_sites": cfg.motif_store.combine_sites,
            "site_kinds": cfg.motif_store.site_kinds,
            "site_window_lengths": cfg.motif_store.site_window_lengths,
            "site_window_center": cfg.motif_store.site_window_center,
            "pwm_window_lengths": cfg.motif_store.pwm_window_lengths,
            "pwm_window_strategy": cfg.motif_store.pwm_window_strategy,
            "min_sites_for_pwm": cfg.motif_store.min_sites_for_pwm,
            "allow_low_sites": cfg.motif_store.allow_low_sites,
            "pseudocounts": cfg.motif_store.pseudocounts,
        },
    }
    signature = sha256_bytes(json.dumps(payload, sort_keys=True).encode("utf-8"))
    return signature, payload


def _find_existing_parse_run(
    *,
    cfg: CruncherConfig,
    config_path: Path,
    tfs: list[str],
    set_index: int,
    signature: str,
) -> Path | None:
    stage_dir = run_group_dir(out_root(config_path, cfg.out_dir), "parse", tfs, set_index)
    if not stage_dir.exists():
        return None
    for child in sorted(stage_dir.iterdir()):
        if not child.is_dir():
            continue
        try:
            manifest = load_manifest(child)
        except FileNotFoundError:
            continue
        if manifest.get("stage") != "parse":
            continue
        if manifest.get("parse_signature") != signature:
            continue
        return child
    return None


def run_parse(cfg: CruncherConfig, config_path: Path) -> None:
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
    catalog_root = resolve_catalog_root(config_path, cfg.motif_store.catalog_root)

    store = _store(cfg, config_path)

    # Prepare output folder
    out_base = out_root(config_path, cfg.out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    groups = regulator_sets(cfg.regulator_sets)
    set_count = len(groups)
    include_set_index = set_count > 1

    lockfile, lockmap = _lockmap_for(cfg, config_path)
    catalog = CatalogIndex.load(catalog_root)
    lock_path = resolve_lock_path(config_path)
    logger.info("Using lockfile: %s", lock_path)
    logger.info(
        "PWM config: source=%s combine_sites=%s site_kinds=%s",
        cfg.motif_store.pwm_source,
        cfg.motif_store.combine_sites,
        cfg.motif_store.site_kinds,
    )

    for set_index, group in enumerate(groups, start=1):
        if not group:
            raise ValueError(f"regulator_sets[{set_index}] is empty.")
        seen: set[str] = set()
        tfs = [tf for tf in group if not (tf in seen or seen.add(tf))]
        signature, signature_payload = _parse_signature(
            cfg=cfg,
            lockfile=lockfile,
            tfs=tfs,
        )
        existing = _find_existing_parse_run(
            cfg=cfg,
            config_path=config_path,
            tfs=tfs,
            set_index=set_index,
            signature=signature,
        )
        if existing is not None:
            logger.info("Parse outputs already generated: %s", existing)
            continue

        run_group = run_group_label(tfs, set_index, include_set_index=include_set_index)
        run_dir = build_run_dir(
            config_path=config_path,
            out_dir=cfg.out_dir,
            stage="parse",
            tfs=tfs,
            set_index=set_index,
            include_set_index=include_set_index,
        )
        ensure_run_dirs(run_dir, meta=True)
        status_writer = RunStatusWriter(
            path=status_path(run_dir),
            stage="parse",
            run_dir=run_dir,
            payload={
                "config_path": str(config_path.resolve()),
                "status_message": "parsing",
                "regulator_set": {"index": set_index, "tfs": tfs, "count": set_count},
                "run_group": run_group,
            },
        )
        update_run_index_from_status(
            config_path,
            run_dir,
            status_writer.payload,
            catalog_root=cfg.motif_store.catalog_root,
        )

        for tf in sorted(tfs):
            locked = lockmap.get(tf)
            if locked is None:
                raise ValueError(f"Missing lock entry for TF '{tf}'")
            ref = MotifRef(source=locked.source, motif_id=locked.motif_id)
            catalog_entry = catalog.entries.get(f"{ref.source}:{ref.motif_id}")
            if catalog_entry is None:
                raise ValueError(f"Catalog entry missing for {ref.source}:{ref.motif_id}")
            provenance = pwm_provenance_summary(
                pwm_source=cfg.motif_store.pwm_source,
                entry=catalog_entry,
                catalog=catalog,
                combine_sites=cfg.motif_store.combine_sites,
                site_kinds=cfg.motif_store.site_kinds,
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
                "sequence_length": cfg.sample.init.length if cfg.sample else None,
                "regulator_set": {"index": set_index, "tfs": tfs, "count": set_count},
                "run_group": run_group,
                "parse_signature": signature,
                "parse_inputs": signature_payload,
            },
        )
        manifest_path = write_manifest(run_dir, manifest)
        update_run_index_from_manifest(
            config_path,
            run_dir,
            manifest,
            catalog_root=cfg.motif_store.catalog_root,
        )
        logger.info("Parse stage complete: %s", run_dir)
        logger.info("Wrote run manifest → %s", manifest_path.relative_to(run_dir.parent))
        status_writer.finish(status="completed", artifacts=[])
        update_run_index_from_status(
            config_path,
            run_dir,
            status_writer.payload,
            catalog_root=cfg.motif_store.catalog_root,
        )
