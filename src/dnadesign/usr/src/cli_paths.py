"""
--------------------------------------------------------------------------------
dnadesign
dnadesign/src/dnadesign/usr/src/cli_paths.py

Path resolution and dataset path contract helpers for the USR CLI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from .dataset import LEGACY_DATASET_PREFIX, RECORDS, Dataset
from .errors import SequencesError

LEGACY_DATASET_PATH_ERROR = (
    "Legacy dataset paths under 'archived/' are not supported. "
    "Use canonical datasets or datasets/_archive/<namespace>/<dataset>."
)


def pkg_usr_root() -> Path:
    """Return the installed dnadesign/usr package directory."""
    return Path(__file__).resolve().parents[1]


def assert_not_legacy_dataset_path(
    path: Path,
    *,
    root: Path | None = None,
    pkg_root: Path | None = None,
) -> None:
    target = Path(path).resolve()
    candidates: list[Path] = []
    if root is not None:
        candidates.append(Path(root).resolve() / LEGACY_DATASET_PREFIX)
    pkg_root = Path(pkg_root).resolve() if pkg_root is not None else pkg_usr_root().resolve()
    candidates.extend(
        [
            pkg_root / LEGACY_DATASET_PREFIX,
            pkg_root / "datasets" / LEGACY_DATASET_PREFIX,
        ]
    )
    for legacy_root in candidates:
        legacy = legacy_root.resolve()
        if target == legacy or target.is_relative_to(legacy):
            raise SequencesError(LEGACY_DATASET_PATH_ERROR)


def assert_supported_root(root: Path, *, pkg_root: Path | None = None) -> None:
    root_resolved = Path(root).resolve()
    pkg_root = Path(pkg_root).resolve() if pkg_root is not None else pkg_usr_root().resolve()
    legacy_roots = {
        (pkg_root / "archived").resolve(),
        (pkg_root / "datasets" / LEGACY_DATASET_PREFIX).resolve(),
    }
    if root_resolved in legacy_roots:
        raise SequencesError(f"Legacy USR root is not supported: {root_resolved}. {LEGACY_DATASET_PATH_ERROR}")


def resolve_path_anywhere(path: Path, *, pkg_root: Path | None = None) -> Path:
    """Resolve CLI paths from cwd first, then relative to the usr package."""
    candidate = Path(path)

    if candidate.is_absolute() and candidate.exists():
        return candidate
    if candidate.exists():
        return candidate

    base = Path(pkg_root).resolve() if pkg_root is not None else pkg_usr_root()
    direct = base / candidate
    if direct.exists():
        return direct

    parts = candidate.parts
    if "dnadesign" in parts and "usr" in parts:
        try:
            idx = parts.index("dnadesign")
            if parts[idx + 1] == "usr":
                sub = Path(*parts[idx + 2 :])
                nested = base / sub
                if nested.exists():
                    return nested
        except (ValueError, IndexError):
            pass

    if parts and parts[0] == "usr":
        short = base / Path(*parts[1:])
        if short.exists():
            return short

    return candidate


def resolve_dataset_for_read(
    root: Path,
    dataset_arg: str,
    *,
    resolve_existing_dataset_id: Callable[[Path, str], str],
    normalize_dataset_id: Callable[[str], str],
    pkg_root: Path | None = None,
) -> Dataset:
    root = Path(root).resolve()
    target = Path(str(dataset_arg)).expanduser()
    if not target.exists():
        ds_name = resolve_existing_dataset_id(root, str(dataset_arg))
        return Dataset(root, ds_name)

    if target.is_file():
        if target.name != RECORDS:
            raise SequencesError(f"Dataset path must point to a dataset directory or '{RECORDS}' file: {target}")
        dataset_dir = target.resolve().parent
    elif target.is_dir():
        dataset_dir = target.resolve()
        if not (dataset_dir / RECORDS).exists():
            raise SequencesError(f"Dataset directory missing '{RECORDS}': {dataset_dir}")
    else:
        raise SequencesError(f"Unsupported dataset path: {target}")

    assert_not_legacy_dataset_path(dataset_dir, root=root, pkg_root=pkg_root)

    meta_path = dataset_dir / "meta.md"
    if not meta_path.exists():
        raise SequencesError(f"Dataset path requires 'meta.md' with a leading 'name:' entry: {dataset_dir}")
    lines = meta_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise SequencesError(f"meta.md is empty: {meta_path}")
    first = lines[0].strip()
    if not first.startswith("name:"):
        raise SequencesError(f"meta.md missing leading 'name:' entry: {meta_path}")
    dataset_name = first.split(":", 1)[1].strip()
    if not dataset_name:
        raise SequencesError(f"meta.md has empty dataset name: {meta_path}")
    if dataset_name == LEGACY_DATASET_PREFIX or dataset_name.startswith(f"{LEGACY_DATASET_PREFIX}/"):
        raise SequencesError(LEGACY_DATASET_PATH_ERROR)
    dataset_name = normalize_dataset_id(dataset_name)

    parts = Path(dataset_name).parts
    dataset_root = dataset_dir
    for _ in range(max(len(parts), 1)):
        dataset_root = dataset_root.parent

    expected_records = dataset_root / Path(dataset_name) / RECORDS
    if expected_records.resolve() != (dataset_dir / RECORDS).resolve():
        raise SequencesError(
            "Dataset path and meta.md name do not align. "
            f"Expected {expected_records}, got {(dataset_dir / RECORDS).resolve()}."
        )
    return Dataset(dataset_root, dataset_name)
