"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/dataset_identity.py

Dataset identity helpers for normalization and path-based dataset opening.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, TypeVar

from .errors import SequencesError

DatasetType = TypeVar("DatasetType")


class DatasetFactory(Protocol[DatasetType]):
    def __call__(self, root: Path, name: str) -> DatasetType: ...


def normalize_dataset_id(name: str, *, legacy_dataset_prefix: str) -> str:
    ds = str(name or "").strip()
    if not ds:
        raise SequencesError("Dataset name cannot be empty.")
    p = Path(ds)
    if p.is_absolute():
        raise SequencesError("Dataset name must be a relative path.")
    if any(part in {".", ".."} for part in p.parts):
        raise SequencesError("Dataset name must not contain '.' or '..'.")
    if p.parts and p.parts[0] == legacy_dataset_prefix:
        raise SequencesError(
            "legacy dataset paths under 'archived/' are not supported. "
            "Use canonical datasets or datasets/_archive/<namespace>/<dataset>."
        )
    return Path(*p.parts).as_posix()


def open_dataset(
    root: Path,
    name_or_path: str,
    *,
    dataset_factory: DatasetFactory[DatasetType],
    records_name: str,
    legacy_dataset_prefix: str,
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
        return dataset_factory(root_path, rel.as_posix())
    return dataset_factory(root_path, normalize_dataset_id(str(name_or_path), legacy_dataset_prefix=legacy_dataset_prefix))
