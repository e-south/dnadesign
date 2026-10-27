"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cluster/src/io/detect.py

Cluster input-source detection.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from pathlib import Path


def _maybe_datasets_root(root: Path, dataset: str) -> Path | None:
    """
    Normalize a candidate USR root to the *datasets root* accepted by our tooling.
    Accepts either:
      - root = /path/to/usr/datasets        -> expects /path/to/usr/datasets/<dataset>/records.parquet
      - root = /path/to/usr                 -> expects /path/to/usr/datasets/<dataset>/records.parquet
      - root = /path/to/datasets            -> expects /path/to/datasets/<dataset>/records.parquet
      - root = /path/with/<dataset>         -> expects /path/with/<dataset>/records.parquet
    Returns the datasets root (e.g., ".../usr/datasets" or ".../datasets") or None if not valid.
    """
    # 1) root is already a datasets root
    p = root / dataset / "records.parquet"
    if p.exists():
        return root
    # 2) root is an "usr" root (contains a datasets/ subdir)
    p = root / "datasets" / dataset / "records.parquet"
    if p.exists():
        return root / "datasets"
    # 3) root is an arbitrary repo root (contains usr/datasets/)
    p = root / "usr" / "datasets" / dataset / "records.parquet"
    if p.exists():
        return root / "usr" / "datasets"
    return None


def detect_context(dataset: str | None, file: str | Path | None, usr_root: str | None = None) -> dict:
    """Detect working context. Returns a dict with keys:
    kind: 'usr'|'parquet'|'csv'
    dataset: str|None
    file: Path|None
    usr_root: Path|None
    """
    if dataset and file:
        raise ValueError("Pass either --dataset or --file, not both.")
    if dataset:
        # 1) Explicit --usr-root (accept both /usr and /usr/datasets)
        if usr_root:
            root = Path(usr_root)
            ds_root = _maybe_datasets_root(root, dataset)
            if ds_root is None:
                raise FileNotFoundError(
                    "USR dataset '{ds}' not found under --usr-root '{rt}'. Tried: {cand1} and {cand2}".format(
                        ds=dataset,
                        rt=str(root),
                        cand1=str(root / "datasets" / dataset / "records.parquet"),
                        cand2=str(root / dataset / "records.parquet"),
                    )
                )
            return {
                "kind": "usr",
                "dataset": dataset,
                "file": ds_root / dataset / "records.parquet",
                "usr_root": ds_root,  # datasets root
            }

        # 2) Environment override (accept both /usr and /usr/datasets)
        env = os.environ.get("DNADESIGN_USR_ROOT")
        if env:
            root = Path(env)
            ds_root = _maybe_datasets_root(root, dataset)
            if ds_root is not None:
                return {
                    "kind": "usr",
                    "dataset": dataset,
                    "file": ds_root / dataset / "records.parquet",
                    "usr_root": ds_root,
                }

        raise FileNotFoundError(
            "Could not resolve USR dataset '{ds}'. Pass --usr-root or set DNADESIGN_USR_ROOT explicitly.".format(
                ds=dataset
            )
        )
    if file:
        p = Path(file)
        if not p.exists():
            raise FileNotFoundError(f"Input file does not exist: {p}")
        if p.suffix.lower() == ".parquet":
            return {
                "kind": "parquet",
                "dataset": None,
                "file": p,
                "usr_root": None,
            }
        if p.suffix.lower() == ".csv":
            return {
                "kind": "csv",
                "dataset": None,
                "file": p,
                "usr_root": None,
            }
        raise ValueError(f"Unsupported file type: {p.suffix}")
    raise FileNotFoundError("Cluster input selection is explicit. Pass --dataset or --file.")
