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


def _search_up_for_datasets_root(dataset: str, start: Path) -> tuple[Path | None, list[str]]:
    """
    Walk upward from `start`, looking for usr/datasets/<dataset>/records.parquet
    or datasets/<dataset>/records.parquet. Return (datasets_root, tried_paths).
    """
    tried: list[str] = []
    for base in [start, *start.parents]:
        # Try usr/datasets and plain datasets under this base
        for sub in ("usr/datasets", "datasets"):
            ds_dir = base / sub / dataset
            tried.append(str(ds_dir / "records.parquet"))
            if (ds_dir / "records.parquet").exists():
                return ds_dir.parent, tried
        # Also try if this base itself is a datasets root
        tried.append(str(base / dataset / "records.parquet"))
        if (base / dataset / "records.parquet").exists():
            return base, tried
    return None, tried


def detect_context(dataset: str | None, file: str | Path | None, usr_root: str | None = None) -> dict:
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
                "cwd_inferred": False,
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
                    "cwd_inferred": False,
                }

        # 3) If CWD is exactly the dataset dir (keep legacy fast path)
        cwd = Path.cwd()
        cwd_records = cwd / "records.parquet"
        if cwd_records.exists() and cwd.name == dataset:
            return {
                "kind": "usr",
                "dataset": dataset,
                "file": cwd_records,
                "usr_root": cwd.parent,  # datasets root
                "cwd_inferred": True,
            }

        # 4) Walk upward from CWD to find usr/datasets/<dataset>/records.parquet (robust)
        ds_root, tried = _search_up_for_datasets_root(dataset, cwd)
        if ds_root is None:
            tried_msg = "\n  - " + "\n  - ".join(tried[:12])
            more = " (…)" if len(tried) > 12 else ""
            raise FileNotFoundError(
                "Could not resolve USR dataset '{ds}'. Search strategy:\n"
                "  1) --usr-root (not provided)\n"
                "  2) DNADESIGN_USR_ROOT (not set or invalid)\n"
                "  3) Walk up from CWD looking for usr/datasets/<dataset>/records.parquet or datasets/<dataset>/records.parquet\n"  # noqa
                "Paths tried:{lst}{more}\n\n"
                "Hint: pass --usr-root, or set DNADESIGN_USR_ROOT to either '/path/to/usr' or '/path/to/usr/datasets'.".format(  # noqa
                    ds=dataset, lst=tried_msg, more=more
                )
            )
        return {
            "kind": "usr",
            "dataset": dataset,
            "file": ds_root / dataset / "records.parquet",
            "usr_root": ds_root,
            "cwd_inferred": True,
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
