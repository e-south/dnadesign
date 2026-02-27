"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/cli_commands/read_parquet_targets.py

Parquet target discovery and interactive selection helpers for read commands.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from ..ui import print_df_plain, render_table_rich


def _list_parquet_candidates(dir_path: Path, glob: str | None = None) -> list[Path]:
    dir_path = Path(dir_path)
    if not dir_path.exists() or not dir_path.is_dir():
        return []
    seen: dict[Path, None] = {}

    def _add(path: Path) -> None:
        if path.exists() and path.is_file() and path.suffix.lower() == ".parquet":
            seen[path.resolve()] = None

    if glob:
        for path in sorted(dir_path.glob(glob)):
            _add(path)
    _add(dir_path / "records.parquet")
    _add(dir_path / "events.parquet")
    for path in sorted(dir_path.glob("events*.parquet")):
        _add(path)
    for path in sorted(dir_path.glob("*.parquet")):
        _add(path)
    candidates = list(seen.keys())
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates


def _human_size(num_bytes: int | None) -> str:
    if not isinstance(num_bytes, int):
        return "?"
    units = ["B", "KB", "MB", "GB", "TB"]
    unit_idx = 0
    value = float(num_bytes)
    while value >= 1024 and unit_idx < len(units) - 1:
        value /= 1024.0
        unit_idx += 1
    return f"{value:.0f}{units[unit_idx]}"


def _prompt_pick_parquet(candidates: list[Path], use_rich: bool) -> Path | None:
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    rows = []
    for idx, path in enumerate(candidates, start=1):
        parquet_file = pq.ParquetFile(str(path))
        rows.append(
            {
                "#": idx,
                "file": path.name,
                "rows": parquet_file.metadata.num_rows,
                "cols": parquet_file.metadata.num_columns,
                "size": _human_size(int(path.stat().st_size)),
            }
        )

    frame = pd.DataFrame(rows, columns=["#", "file", "rows", "cols", "size"])
    message = "Multiple Parquet files found. Choose one by number (Enter = newest, q = abort):"
    if use_rich:
        render_table_rich(frame, title="Pick a Parquet file", caption=message)
    else:
        print_df_plain(frame)
        print(message)
    selected = input("> ").strip().lower()
    if selected in {"q", "quit", "n"}:
        print("Aborted.")
        return None
    if not selected:
        return candidates[0]
    try:
        index = int(selected)
        if 1 <= index <= len(candidates):
            return candidates[index - 1]
    except ValueError:
        pass
    print("Invalid selection. Aborted.")
    return None


def _select_parquet_target_interactive(
    path_like: Path,
    glob: str | None,
    use_rich: bool,
    *,
    deps,
    root: Path | None = None,
    confirm_if_inferred: bool = False,
) -> Path | None:
    _ = confirm_if_inferred
    path = Path(path_like)
    if path.exists():
        deps.assert_not_legacy_dataset_path(path, root)
    if path.is_file() and path.suffix.lower() == ".parquet":
        return path
    if path.is_dir():
        candidates = _list_parquet_candidates(path, glob=glob)
        if not candidates:
            print(f"(no Parquet files found under {path})")
            print(
                "Tip: cd into a dataset folder (with records.parquet) or pass a dataset name (e.g., 'usr cols demo')."
            )
            return None
        if len(candidates) == 1:
            return candidates[0]
        return _prompt_pick_parquet(candidates, use_rich)
    return None


def _resolve_parquet_from_dir(dir_path: Path, glob: str | None = None) -> Path:
    dir_path = Path(dir_path)
    if not dir_path.exists() or not dir_path.is_dir():
        raise FileNotFoundError(f"{dir_path} is not a directory")

    candidates = []
    if (dir_path / "events.parquet").exists():
        return dir_path / "events.parquet"
    if glob:
        candidates = sorted(dir_path.glob(glob))
    if not candidates:
        candidates = sorted(dir_path.glob("events*.parquet"))
    if not candidates and (dir_path / "records.parquet").exists():
        return dir_path / "records.parquet"
    if not candidates:
        candidates = sorted(dir_path.glob("*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No Parquet files found under {dir_path}")
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def _resolve_parquet_target(path_like: Path, glob: str | None = None) -> Path:
    path = Path(path_like)
    if path.is_file() and path.suffix.lower() == ".parquet":
        return path
    if path.is_dir():
        return _resolve_parquet_from_dir(path, glob=glob)
    raise FileNotFoundError(f"Target not found: {path}")
