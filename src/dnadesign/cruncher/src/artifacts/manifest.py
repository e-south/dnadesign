"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/artifacts/manifest.py

Read and write run manifest artifacts for Cruncher runs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from dnadesign.cruncher.artifacts.atomic_write import atomic_write_json
from dnadesign.cruncher.artifacts.layout import manifest_path
from dnadesign.cruncher.config.schema_v3 import CruncherConfig
from dnadesign.cruncher.store.catalog_index import CatalogEntry, CatalogIndex
from dnadesign.cruncher.store.lockfile import LockedMotif
from dnadesign.cruncher.utils.hashing import sha256_path
from dnadesign.cruncher.utils.paths import resolve_catalog_root


def _motif_payload(
    *,
    tf_name: str,
    locked: LockedMotif,
    entry: CatalogEntry,
) -> Dict[str, Any]:
    return {
        "tf_name": tf_name,
        "source": locked.source,
        "motif_id": locked.motif_id,
        "sha256": locked.sha256,
        "dataset_id": locked.dataset_id,
        "has_matrix": entry.has_matrix,
        "has_sites": entry.has_sites,
        "site_count": entry.site_count,
        "site_total": entry.site_total,
        "site_kind": entry.site_kind,
        "dataset_source": entry.dataset_source,
        "dataset_method": entry.dataset_method,
        "reference_genome": entry.reference_genome,
        "matrix_length": entry.matrix_length,
        "matrix_source": entry.matrix_source,
        "matrix_semantics": entry.matrix_semantics,
        "tags": entry.tags,
    }


def build_run_manifest(
    *,
    stage: str,
    cfg: CruncherConfig,
    config_path: Path,
    lock_path: Path,
    lockmap: Dict[str, LockedMotif],
    catalog: CatalogIndex,
    run_dir: Path,
    artifacts: Optional[Iterable[object]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    motifs: List[Dict[str, Any]] = []
    for tf_name, locked in sorted(lockmap.items()):
        key = f"{locked.source}:{locked.motif_id}"
        entry = catalog.entries.get(key)
        if entry is None:
            raise ValueError(f"Locked motif {key} for TF '{tf_name}' not found in catalog")
        motifs.append(_motif_payload(tf_name=tf_name, locked=locked, entry=entry))

    manifest = {
        "stage": stage,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir.resolve()),
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_path(config_path),
        "lockfile_path": str(lock_path.resolve()),
        "lockfile_sha256": sha256_path(lock_path),
        "motif_store": {
            "catalog_root": str(resolve_catalog_root(config_path, cfg.catalog.catalog_root)),
            "pwm_source": cfg.catalog.pwm_source,
            "site_kinds": cfg.catalog.site_kinds,
            "combine_sites": cfg.catalog.combine_sites,
            "min_sites_for_pwm": cfg.catalog.min_sites_for_pwm,
        },
        "motifs": motifs,
        "artifacts": list(artifacts or []),
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_manifest(run_dir: Path, manifest: Dict[str, Any]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_path(run_dir)
    atomic_write_json(path, manifest)
    return path


def load_manifest(run_dir: Path) -> Dict[str, Any]:
    path = manifest_path(run_dir)
    if not path.exists():
        meta_dir = path.parent
        existing = []
        if meta_dir.exists() and meta_dir.is_dir():
            existing = sorted([p.name for p in meta_dir.iterdir() if p.is_file()])
        hint = (
            "meta/run_manifest.json not found. This often means the sample run was interrupted "
            "before metadata was written."
        )
        if existing:
            hint += f" Existing meta files: {', '.join(existing[:10])}."
        hint += " Re-run sampling with `cruncher sample -c <CONFIG>`."
        raise FileNotFoundError(f"{hint} (run: {run_dir})")
    return json.loads(path.read_text())
