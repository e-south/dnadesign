"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cluster/src/runs/store.py

Cluster run-store helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..layout import explicit_results_root
from .contracts import RunIndexEntry


def runs_root(root: Path | str | None = None) -> Path:
    root = explicit_results_root(root)
    root.mkdir(parents=True, exist_ok=True)
    # Ensure index file exists
    idx = root / "index.parquet"
    if not idx.exists():
        pd.DataFrame(columns=RunIndexEntry.columns()).to_parquet(idx, index=False)
    return root


def alias_dir(root: Path, alias: str) -> Path:
    d = root / alias
    d.mkdir(parents=True, exist_ok=True)
    return d


def fit_run_dir(root: Path, alias: str, slug: str) -> Path:
    d = alias_dir(root, alias) / "fits" / slug
    d.mkdir(parents=True, exist_ok=False)
    return d


def umap_run_dir(root: Path, alias: str, slug: str) -> Path:
    d = alias_dir(root, alias) / "umap" / slug
    d.mkdir(parents=True, exist_ok=False)
    return d


def analysis_run_dir(root: Path, alias: str, slug: str) -> Path:
    d = alias_dir(root, alias) / "analysis" / slug
    d.mkdir(parents=True, exist_ok=False)
    return d


def sweep_run_dir(root: Path, alias: str, slug: str) -> Path:
    d = alias_dir(root, alias) / "sweeps" / slug
    d.mkdir(parents=True, exist_ok=False)
    return d


def write_run_meta(run_dir: Path, meta: dict) -> Path:
    return _write_json_artifact(run_dir / "run.json", meta)


def write_labels(run_dir: Path, labels_df: pd.DataFrame) -> Path:
    p = run_dir / "labels.parquet"
    labels_df.to_parquet(p, index=False)
    return p


def write_summary(run_dir: Path, summary: dict) -> Path:
    return _write_json_artifact(run_dir / "summary.json", summary)


def write_log(run_dir: Path, event: dict) -> None:
    p = run_dir / "log.jsonl"
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, sort_keys=True) + "\n")


def append_records_md(run_dir: Path, markdown: str) -> Path:
    """
    Append a Markdown entry describing a command invocation to <run_dir>/records.md.
    Creates the file with a header if it doesn't exist yet.
    """
    p = run_dir / "records.md"
    if not p.exists():
        p.write_text("# Run records\n\n", encoding="utf-8")
    # Always insert a trailing newline between entries
    with p.open("a", encoding="utf-8") as f:
        # normalize to avoid duplicate blank lines
        text = markdown.rstrip() + "\n\n"
        f.write(text)
    return p


def write_umap_meta(umap_dir_path: Path, meta: dict) -> Path:
    return _write_json_artifact(umap_dir_path / "umap.json", meta)


def write_analysis_meta(analysis_dir_path: Path, meta: dict) -> Path:
    return _write_json_artifact(analysis_dir_path / "analysis.json", meta)


def write_sweep_meta(sweep_dir_path: Path, meta: dict) -> Path:
    return _write_json_artifact(sweep_dir_path / "sweep.json", meta)


def write_umap_coords(umap_dir_path: Path, coords_df: pd.DataFrame) -> Path:
    p = umap_dir_path / "coords.parquet"
    coords_df.to_parquet(p, index=False)
    return p


def _write_json_artifact(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path
