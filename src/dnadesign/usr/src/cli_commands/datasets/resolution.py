"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/cli_commands/datasets/resolution.py

Dataset-id normalization and interactive selection helpers for USR CLI commands.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from ...cli_support.rendering import print_df_plain, render_table_rich
from ...dataset import ARCHIVE_DATASET_PREFIX, normalize_dataset_id
from ...datasets.identity import ARCHIVED_DATASET_ID_ERROR
from ...errors import SequencesError
from .catalog import list_datasets


def resolve_existing_dataset_id(root: Path, dataset: str) -> str:
    root = Path(root).resolve()
    dataset_id = _normalize_dataset_id(dataset)
    if dataset_id == ARCHIVE_DATASET_PREFIX or dataset_id.startswith(f"{ARCHIVE_DATASET_PREFIX}/"):
        raise SystemExit(ARCHIVED_DATASET_ID_ERROR)
    all_datasets = list_datasets(root)
    if "/" in dataset_id:
        if dataset_id not in all_datasets:
            raise SystemExit(f"Dataset not found: {dataset_id}")
        return dataset_id
    candidates = [name for name in all_datasets if name.split("/", 1)[-1] == dataset_id]
    if dataset_id in all_datasets and len(candidates) == 1:
        return dataset_id
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise SystemExit(f"Dataset not found: {dataset_id}")
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
        return _prompt_pick_dataset(root, list_datasets(root), use_rich)
    path = cwd
    for _ in range(4):
        inferred = _dataset_id_from_path(root, path)
        if inferred:
            return inferred
        path = path.parent
    print(
        "Dataset not provided and could not be inferred from CWD. "
        "Run inside a dataset folder under --root or pass a dataset name."
    )
    return None


def _normalize_dataset_id(dataset: str) -> str:
    try:
        return normalize_dataset_id(dataset)
    except SequencesError as exc:
        raise SystemExit(str(exc)) from None


def _dataset_exists(root: Path, dataset_id: str) -> bool:
    return (root / Path(dataset_id) / "records.parquet").exists()


def _dataset_id_from_path(root: Path, path: Path) -> str | None:
    root = Path(root).resolve()
    path = Path(path).resolve()
    try:
        relative = path.relative_to(root)
    except ValueError:
        return None
    if relative.parts and relative.parts[0] == ARCHIVE_DATASET_PREFIX:
        return None
    if len(relative.parts) >= 2:
        candidate = Path(relative.parts[0], relative.parts[1])
        if _dataset_exists(root, candidate.as_posix()):
            return candidate.as_posix()
    if len(relative.parts) >= 1:
        candidate = Path(relative.parts[0])
        if _dataset_exists(root, candidate.as_posix()):
            return candidate.as_posix()
    return None


def _prompt_pick_dataset(root: Path, names: list[str], use_rich: bool) -> str | None:
    if not names:
        print(f"(no datasets under {root})")
        return None
    if len(names) == 1:
        return names[0]
    rows = []
    for index, name in enumerate(names, start=1):
        records_path = root / name / "records.parquet"
        parquet = pq.ParquetFile(str(records_path))
        rows.append(
            {
                "#": index,
                "dataset": name,
                "rows": parquet.metadata.num_rows,
                "cols": parquet.metadata.num_columns,
            }
        )
    frame = pd.DataFrame(rows, columns=["#", "dataset", "rows", "cols"])
    message = "Multiple datasets found. Choose one by number (Enter = first, q = abort):"
    if use_rich:
        render_table_rich(frame, title="Pick a dataset", caption=str(root))
    else:
        print_df_plain(frame)
        print(message)
    selection = input("> ").strip().lower()
    if selection in {"q", "quit", "n"}:
        print("Aborted.")
        return None
    if not selection:
        return names[0]
    try:
        choice = int(selection)
        if 1 <= choice <= len(names):
            return names[choice - 1]
    except ValueError:
        pass
    print("Invalid selection. Aborted.")
    return None
