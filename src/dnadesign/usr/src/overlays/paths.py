"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/overlays/paths.py

Overlay path and directory enumeration helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path


def is_temporary_overlay_entry(entry: Path) -> bool:
    return entry.name.endswith(".tmp.parquet")


def derived_dir(dataset_dir: Path, *, derived_dir_name: str) -> Path:
    return Path(dataset_dir) / derived_dir_name


def overlay_path(dataset_dir: Path, namespace: str, *, derived_dir_name: str) -> Path:
    return derived_dir(dataset_dir, derived_dir_name=derived_dir_name) / f"{namespace}.parquet"


def overlay_dir_path(dataset_dir: Path, namespace: str, *, derived_dir_name: str) -> Path:
    return derived_dir(dataset_dir, derived_dir_name=derived_dir_name) / str(namespace)


def list_overlays(
    dataset_dir: Path,
    *,
    derived_dir_name: str,
    overlay_parts,
    list_cache: dict[str, tuple[tuple[tuple[str, bool, int, int], ...], tuple[str, ...]]],
    list_cache_max: int,
) -> list[Path]:
    overlays_dir = derived_dir(dataset_dir, derived_dir_name=derived_dir_name)
    if not overlays_dir.exists():
        return []
    if not overlays_dir.is_dir():
        return []

    entry_signatures: list[tuple[str, bool, int, int]] = []
    for entry in overlays_dir.iterdir():
        try:
            stat = entry.stat()
        except FileNotFoundError:
            continue
        entry_signatures.append((entry.name, entry.is_dir(), int(stat.st_mtime_ns), int(stat.st_size)))
    signature = tuple(sorted(entry_signatures, key=lambda item: item[0]))

    cache_key = str(overlays_dir.absolute())
    cached = list_cache.get(cache_key)
    if cached is not None and cached[0] == signature:
        return [Path(path) for path in cached[1]]

    overlays: list[Path] = []
    for name, is_dir, _mtime_ns, _size in signature:
        entry = overlays_dir / name
        if is_temporary_overlay_entry(entry):
            continue
        if not is_dir and entry.suffix == ".parquet":
            overlays.append(entry)
            continue
        if is_dir and overlay_parts(entry):
            overlays.append(entry)
    overlays_sorted = sorted(overlays, key=lambda path: path.name)

    list_cache[cache_key] = (signature, tuple(str(path) for path in overlays_sorted))
    if len(list_cache) > list_cache_max:
        list_cache.clear()
    return overlays_sorted


def overlay_parts(
    path: Path,
    *,
    part_prefix: str,
    parts_cache: dict[str, tuple[int, int, tuple[str, ...]]],
    parts_cache_max: int,
) -> list[Path]:
    overlay_path_value = Path(path)
    if overlay_path_value.is_dir():
        try:
            stat = overlay_path_value.stat()
        except FileNotFoundError:
            return []
        cache_key = str(overlay_path_value.absolute())
        stat_key = (int(stat.st_mtime_ns), int(stat.st_size))
        cached = parts_cache.get(cache_key)
        if cached is not None and cached[0] == stat_key[0] and cached[1] == stat_key[1]:
            return [Path(part_path) for part_path in cached[2]]
        parts = tuple(str(part) for part in sorted(overlay_path_value.glob(f"{part_prefix}*.parquet")))
        parts_cache[cache_key] = (stat_key[0], stat_key[1], parts)
        if len(parts_cache) > parts_cache_max:
            parts_cache.clear()
        return [Path(part_path) for part_path in parts]
    if overlay_path_value.is_file():
        return [overlay_path_value]
    return []
