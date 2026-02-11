"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/cli_commands/datasets.py

Dataset discovery and resolution helpers for USR CLI commands.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from ..dataset import LEGACY_DATASET_PREFIX, normalize_dataset_id
from ..errors import SequencesError
from ..ui import print_df_plain, render_table_rich


def list_datasets(root: Path) -> list[str]:
    root = root.resolve()
    if not root.exists():
        return []
    names: set[str] = set()
    for p in root.iterdir():
        if not p.is_dir():
            continue
        if p.name == LEGACY_DATASET_PREFIX:
            continue
        if (p / "records.parquet").exists():
            names.add(p.name)
            continue
        for child in p.iterdir():
            if child.is_dir() and (child / "records.parquet").exists():
                names.add(f"{p.name}/{child.name}")
    return sorted(names)


def resolve_existing_dataset_id(root: Path, dataset: str) -> str:
    root = Path(root).resolve()
    ds = _normalize_dataset_id(dataset)
    if ds == LEGACY_DATASET_PREFIX or ds.startswith(f"{LEGACY_DATASET_PREFIX}/"):
        raise SystemExit(
            "Legacy dataset paths under 'archived/' are not supported. "
            "Use canonical datasets or datasets/_archive/<namespace>/<dataset>."
        )
    all_ds = list_datasets(root)
    if "/" in ds:
        if ds not in all_ds:
            raise SystemExit(f"Dataset not found: {ds}")
        return ds
    candidates = [name for name in all_ds if name.split("/", 1)[-1] == ds]
    if ds in all_ds and len(candidates) == 1:
        return ds
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise SystemExit(f"Dataset not found: {ds}")
    raise SystemExit("Ambiguous dataset name. Use a namespace-qualified id. Matches: " + ", ".join(sorted(candidates)))


def resolve_dataset_name_interactive(root: Path, dataset: str | None, use_rich: bool) -> str | None:
    """
    If dataset is None, try to infer from CWD:
      - If CWD is <root>/<dataset>[/...], use that dataset
      - If CWD is <root>/<namespace>/<dataset>[/...], use that dataset
      - If CWD == <root>, prompt to pick a dataset
    """
    root = Path(root).resolve()
    if dataset:
        return resolve_existing_dataset_id(root, dataset)
    cwd = Path.cwd().resolve()
    inferred = _dataset_id_from_path(root, cwd)
    if inferred:
        return inferred
    if cwd == root:
        from_names = list_datasets(root)
        return _prompt_pick_dataset(root, from_names, use_rich)
    p = cwd
    for _ in range(4):
        inferred = _dataset_id_from_path(root, p)
        if inferred:
            return inferred
        p = p.parent
    print(
        "Dataset not provided and could not be inferred from CWD. Run inside a dataset folder under --root or pass a dataset name."  # noqa
    )
    return None


def _normalize_dataset_id(dataset: str) -> str:
    try:
        return normalize_dataset_id(dataset)
    except SequencesError as e:
        raise SystemExit(str(e)) from None


def _dataset_exists(root: Path, dataset_id: str) -> bool:
    return (root / Path(dataset_id) / "records.parquet").exists()


def _dataset_id_from_path(root: Path, path: Path) -> str | None:
    root = Path(root).resolve()
    p = Path(path).resolve()
    try:
        rel = p.relative_to(root)
    except ValueError:
        return None
    if rel.parts and rel.parts[0] == LEGACY_DATASET_PREFIX:
        raise SystemExit(
            "Legacy dataset paths under 'archived/' are not supported. "
            "Use canonical datasets or datasets/_archive/<namespace>/<dataset>."
        )
    if len(rel.parts) >= 2:
        cand = Path(rel.parts[0], rel.parts[1])
        if _dataset_exists(root, cand.as_posix()):
            return cand.as_posix()
    if len(rel.parts) >= 1:
        cand = Path(rel.parts[0])
        if _dataset_exists(root, cand.as_posix()):
            return cand.as_posix()
    return None


def _prompt_pick_dataset(root: Path, names: list[str], use_rich: bool) -> str | None:
    if not names:
        print(f"(no datasets under {root})")
        return None
    if len(names) == 1:
        return names[0]
    rows = []
    for idx, name in enumerate(names, start=1):
        rp = root / name / "records.parquet"
        pf = pq.ParquetFile(str(rp))
        rows.append(
            {
                "#": idx,
                "dataset": name,
                "rows": pf.metadata.num_rows,
                "cols": pf.metadata.num_columns,
            }
        )
    df = pd.DataFrame(rows, columns=["#", "dataset", "rows", "cols"])
    msg = "Multiple datasets found. Choose one by number (Enter = first, q = abort):"
    if use_rich:
        render_table_rich(df, title="Pick a dataset", caption=str(root))
    else:
        print_df_plain(df)
        print(msg)
    sel = input("> ").strip().lower()
    if sel in {"q", "quit", "n"}:
        print("Aborted.")
        return None
    if not sel:
        return names[0]
    try:
        k = int(sel)
        if 1 <= k <= len(names):
            return names[k - 1]
    except ValueError:
        pass
    print("Invalid selection. Aborted.")
    return None
