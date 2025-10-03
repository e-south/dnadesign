"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/cluster/src/io/detect.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from pathlib import Path


def detect_context(
    dataset: str | None, file: str | Path | None, usr_root: str | None = None
) -> dict:
    """Detect working context. Returns a dict with keys:
    kind: 'usr'|'parquet'|'csv'
    dataset: str|None
    file: Path|None
    usr_root: Path|None
    cwd_inferred: bool
    """
    if dataset and file:
        raise ValueError("Pass either --dataset or --file, not both.")
    if dataset:
        # 1) If caller gave an explicit usr_root, use it directly.
        if usr_root:
            root = Path(usr_root)
            ds_dir = root / dataset
            file_path = ds_dir / "records.parquet"
            if not file_path.exists():
                raise FileNotFoundError(
                    f"USR dataset '{dataset}' not found under '{root}'. Expected {file_path}"
                )
            return {
                "kind": "usr",
                "dataset": dataset,
                "file": file_path,
                "usr_root": root,
                "cwd_inferred": False,
            }

        # 2) If no usr_root, allow environment override; else start from CWD.
        root_candidate = Path(os.environ.get("DNADESIGN_USR_ROOT", Path.cwd()))

        # Special case: if we're already **inside** the dataset directory
        # (i.e., CWD contains records.parquet and its name matches `dataset`),
        # then the USR root is the parent and the records live here.
        cwd = Path.cwd()
        cwd_records = cwd / "records.parquet"
        if cwd_records.exists() and cwd.name == dataset:
            return {
                "kind": "usr",
                "dataset": dataset,
                "file": cwd_records,
                "usr_root": cwd.parent,  # the datasets/ parent
                "cwd_inferred": True,
            }

        # 3) Otherwise assume <root_candidate>/<dataset>/records.parquet
        root = root_candidate if root_candidate.exists() else Path.cwd()
        ds_dir = root / dataset
        file_path = ds_dir / "records.parquet"
        if not file_path.exists():
            raise FileNotFoundError(
                f"USR dataset '{dataset}' not found under '{root}'. Expected {file_path}"
            )
        return {
            "kind": "usr",
            "dataset": dataset,
            "file": file_path,
            "usr_root": root,
            "cwd_inferred": False,
        }
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
                "cwd_inferred": False,
            }
        if p.suffix.lower() == ".csv":
            return {
                "kind": "csv",
                "dataset": None,
                "file": p,
                "usr_root": None,
                "cwd_inferred": False,
            }
        raise ValueError(f"Unsupported file type: {p.suffix}")
    # Auto-detect in CWD
    cwd = Path.cwd()
    rp = cwd / "records.parquet"
    if rp.exists():
        # Heuristic: if parent looks like usr/datasets/<dataset>/
        parts = rp.resolve().parts
        if len(parts) >= 3 and parts[-3] == "datasets":
            dataset = parts[-2]
            usr_root = Path(*parts[:-2])
            return {
                "kind": "usr",
                "dataset": dataset,
                "file": rp,
                "usr_root": usr_root,
                "cwd_inferred": True,
            }
        return {
            "kind": "parquet",
            "dataset": None,
            "file": rp,
            "usr_root": None,
            "cwd_inferred": True,
        }
    raise FileNotFoundError(
        "Could not infer context. Pass --dataset or --file, or run in a folder with records.parquet."
    )
