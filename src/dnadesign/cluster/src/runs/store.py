"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/cluster/src/runs/store.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

DEFAULT_ENV_KEY = "DNADESIGN_CLUSTER_RUNS_DIR"
DEFAULT_DIRNAME = "cluster_log"
FLAT_UMAP_ENV = "DNADESIGN_CLUSTER_FLAT_UMAP_DIRS"  # "1", "true", "yes" => flat layout



def _package_cluster_dir() -> Path:
    """
    Resolve the installed package's 'cluster' directory:
    .../dnadesign/cluster/ (sibling of 'cluster/src').
    """
    return Path(__file__).resolve().parents[2]


def runs_root(default_base: Path | None = None) -> Path:
    env = os.environ.get(DEFAULT_ENV_KEY)
    if env:
        root = Path(env)
    else:
        # Default: <package_cluster_dir>/cluster_log  (sibling to cluster/src)
        cluster_dir = _package_cluster_dir()
        base = default_base or cluster_dir
        new_root = base / DEFAULT_DIRNAME
        old_root = base / "batch_results"
        # One-time migration: move previous default into the new standardized name.
        if old_root.exists() and not new_root.exists():
            try:
                old_root.rename(new_root)
            except Exception:
                # If rename fails, keep using old_root rather than duplicating.
                new_root = old_root
        root = new_root
    root.mkdir(parents=True, exist_ok=True)
    # Ensure index file exists
    idx = root / "index.parquet"
    if not idx.exists():
        pd.DataFrame(
            columns=[
                "kind",
                "run_slug",
                "alias",
                "created_utc",
                "source_kind",
                "source_ref",
                "x_col",
                "n_rows",
                "n_clusters",
                "algo",
                "algo_params",
                "input_sig_hash",
                "labels_path",
                "status",
                "umap_slug",
                "umap_params",
                "coords_path",
                "plot_paths",
            ]
        ).to_parquet(idx, index=False)
    return root


def create_run_dir(root: Path, slug: str) -> Path:
    d = root / slug
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_run_meta(run_dir: Path, meta: dict) -> Path:
    p = run_dir / "run.json"
    p.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return p


def write_labels(run_dir: Path, labels_df: pd.DataFrame) -> Path:
    p = run_dir / "labels.parquet"
    labels_df.to_parquet(p, index=False)
    return p


def write_summary(run_dir: Path, summary: dict) -> Path:
    p = run_dir / "summary.json"
    p.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return p


def write_log(run_dir: Path, event: dict) -> None:
    p = run_dir / "log.jsonl"
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, sort_keys=True) + "\n")


# ---------------- UMAP helpers ----------------
def umap_dir(run_dir: Path, umap_slug: str) -> Path:
    flat = str(os.environ.get(FLAT_UMAP_ENV, "")).strip().lower() in {"1", "true", "yes", "y", "on"}
    d = (run_dir / umap_slug) if flat else (run_dir / "umap" / umap_slug)
    d.mkdir(parents=True, exist_ok=True)
    (d / "plots").mkdir(parents=True, exist_ok=True)
    return d


def write_umap_meta(umap_dir_path: Path, meta: dict) -> Path:
    p = umap_dir_path / "umap.json"
    p.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return p


def write_umap_coords(umap_dir_path: Path, coords_df: pd.DataFrame) -> Path:
    p = umap_dir_path / "coords.parquet"
    coords_df.to_parquet(p, index=False)
    return p
