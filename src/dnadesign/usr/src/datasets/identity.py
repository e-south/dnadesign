"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/datasets/identity.py

Dataset identity helpers for normalization and path-based dataset opening.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, TypeVar

from ..errors import SequencesError

DatasetType = TypeVar("DatasetType")
ARCHIVED_DATASET_ID_ERROR = (
    "Dataset ids under 'archived/' are reserved for archived storage. "
    "Use a live dataset id, or pass an explicit path under datasets/archived/<dataset> "
    "when you need archived material."
)


class DatasetFactory(Protocol[DatasetType]):
    def __call__(self, root: Path, name: str) -> DatasetType: ...


def _read_dataset_name_from_meta(dataset_dir: Path) -> str:
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
    return dataset_name


def normalize_dataset_id(name: str, *, archive_dataset_prefix: str) -> str:
    ds = str(name or "").strip()
    if not ds:
        raise SequencesError("Dataset name cannot be empty.")
    p = Path(ds)
    if p.is_absolute():
        raise SequencesError("Dataset name must be a relative path.")
    if any(part in {".", ".."} for part in p.parts):
        raise SequencesError("Dataset name must not contain '.' or '..'.")
    if p.parts and p.parts[0] == archive_dataset_prefix:
        raise SequencesError(ARCHIVED_DATASET_ID_ERROR)
    return Path(*p.parts).as_posix()


def open_dataset(
    root: Path,
    name_or_path: str,
    *,
    dataset_factory: DatasetFactory[DatasetType],
    records_name: str,
    archive_dataset_prefix: str,
) -> DatasetType:
    root_path = Path(root).resolve()
    target = Path(str(name_or_path)).expanduser()
    if target.exists():
        if target.is_file() and target.name == records_name:
            dataset_dir = target.parent
        elif target.is_dir() and (target / records_name).exists():
            dataset_dir = target
        else:
            raise SequencesError(f"Path does not point to a dataset: {target}")
        try:
            rel = dataset_dir.resolve().relative_to(root_path)
        except ValueError as error:
            raise SequencesError(f"Dataset path must live under root: {root_path}") from error
        if rel.parts and rel.parts[0] == archive_dataset_prefix:
            dataset_name = normalize_dataset_id(
                _read_dataset_name_from_meta(dataset_dir),
                archive_dataset_prefix=archive_dataset_prefix,
            )
            return dataset_factory(dataset_dir.parent, dataset_name)
        return dataset_factory(root_path, rel.as_posix())
    return dataset_factory(
        root_path,
        normalize_dataset_id(
            str(name_or_path),
            archive_dataset_prefix=archive_dataset_prefix,
        ),
    )
