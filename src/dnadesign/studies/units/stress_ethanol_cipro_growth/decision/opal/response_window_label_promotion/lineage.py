"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_window_label_promotion/lineage.py

Serialize label promotion publication and maintain its authoritative lineage head.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import fcntl
import json
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile
from typing import Iterator

from dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations.aggregation import (
    VALUE_COLUMNS,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations.artifact_io import (
    file_sha256,
    read_json_object,
)

from .contracts import PROMOTION_FILENAME, ResponseWindowLabelPromotionError

LINEAGE_HEAD_FILENAME = "response_window_label_promotion.head.json"
LINEAGE_LOCK_FILENAME = ".response_window_label_promotion.lock"
LINEAGE_SCHEMA_ID = "stress_ethanol_cipro_growth.response_window_label_promotion_head.v1"
_HEAD_FIELDS = {
    "schema_id",
    "manifest_path",
    "manifest_sha256",
    "label_event_count",
    "unique_candidate_count",
    "max_observed_round",
}


@dataclass(frozen=True)
class LineageHead:
    manifest_path: str
    manifest_sha256: str
    label_event_count: int
    unique_candidate_count: int
    max_observed_round: int


@contextmanager
def lineage_publication_lock(dataset_root: Path) -> Iterator[None]:
    """Hold one dataset-scoped process lock through publication and head update."""

    opal_dir = _opal_directory(dataset_root)
    lock_path = opal_dir / LINEAGE_LOCK_FILENAME
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def load_lineage_head(dataset_root: Path) -> LineageHead | None:
    root = Path(dataset_root).resolve()
    head_path = _opal_directory(root) / LINEAGE_HEAD_FILENAME
    if not head_path.exists():
        return None
    try:
        payload = read_json_object(head_path, label="response-window promotion lineage head")
    except (OSError, UnicodeError, ValueError) as exc:
        raise ResponseWindowLabelPromotionError(f"could not read promotion lineage head: {exc}") from exc
    if set(payload) != _HEAD_FIELDS or payload["schema_id"] != LINEAGE_SCHEMA_ID:
        raise ResponseWindowLabelPromotionError("promotion lineage head identity is malformed.")
    manifest_path = _confined_manifest_path(payload["manifest_path"], root=root)
    manifest_sha256 = payload["manifest_sha256"]
    if not isinstance(manifest_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", manifest_sha256) is None:
        raise ResponseWindowLabelPromotionError("promotion lineage head manifest digest is malformed.")
    if not manifest_path.is_file() or file_sha256(manifest_path) != manifest_sha256:
        raise ResponseWindowLabelPromotionError("promotion lineage head manifest digest disagrees.")
    counts = {
        field: _nonnegative_integer(payload[field], field=field)
        for field in ("label_event_count", "unique_candidate_count", "max_observed_round")
    }
    if counts["label_event_count"] < 1 or counts["unique_candidate_count"] < 1:
        raise ResponseWindowLabelPromotionError("promotion lineage head counts must be positive.")
    from .publication import verify_label_bundle

    snapshot = verify_label_bundle(
        root,
        relative_dir=PurePosixPath(manifest_path.relative_to(root).as_posix()).parent,
        expected_width=len(VALUE_COLUMNS),
    )
    actual = {
        "label_event_count": snapshot.promotion.row_count,
        "unique_candidate_count": int(snapshot.labels["id"].astype(str).nunique()),
        "max_observed_round": int(snapshot.labels["r"].astype(int).max()),
    }
    if counts != actual:
        raise ResponseWindowLabelPromotionError("promotion lineage head inventory disagrees with its label artifact.")
    return LineageHead(
        manifest_path=manifest_path.relative_to(root).as_posix(),
        manifest_sha256=manifest_sha256,
        **counts,
    )


def require_current_parent(
    *,
    head: LineageHead | None,
    prior_reference: dict[str, object] | None,
    incoming_round: int,
) -> None:
    """Reject no-parent later rounds and parents that are no longer the head."""

    if head is None:
        if prior_reference is not None or incoming_round != 0:
            raise ResponseWindowLabelPromotionError(
                "a lineage genesis requires no prior promotion and observed round 0."
            )
        return
    if prior_reference is None:
        raise ResponseWindowLabelPromotionError(
            "the promotion lineage already has a head; the next publication must name that prior manifest."
        )
    if (
        prior_reference.get("manifest_path") != head.manifest_path
        or prior_reference.get("manifest_sha256") != head.manifest_sha256
        or prior_reference.get("label_event_count") != head.label_event_count
        or prior_reference.get("unique_candidate_count") != head.unique_candidate_count
        or prior_reference.get("max_observed_round") != head.max_observed_round
    ):
        raise ResponseWindowLabelPromotionError(
            "the requested prior promotion is stale and does not match the authoritative lineage head."
        )


def update_lineage_head(
    dataset_root: Path,
    *,
    manifest_path: Path,
    label_event_count: int,
    unique_candidate_count: int,
    max_observed_round: int,
) -> None:
    """Atomically replace the lineage head while the publication lock is held."""

    root = Path(dataset_root).resolve()
    manifest = Path(manifest_path).resolve()
    try:
        relative = manifest.relative_to(root)
    except ValueError as exc:
        raise ResponseWindowLabelPromotionError("promotion manifest is outside the lineage dataset root.") from exc
    payload = {
        "schema_id": LINEAGE_SCHEMA_ID,
        "manifest_path": relative.as_posix(),
        "manifest_sha256": file_sha256(manifest),
        "label_event_count": int(label_event_count),
        "unique_candidate_count": int(unique_candidate_count),
        "max_observed_round": int(max_observed_round),
    }
    opal_dir = _opal_directory(root)
    head_path = opal_dir / LINEAGE_HEAD_FILENAME
    with NamedTemporaryFile("w", encoding="utf-8", dir=opal_dir, prefix=".lineage-head-", delete=False) as handle:
        temporary_path = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary_path, head_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _opal_directory(dataset_root: Path) -> Path:
    root = Path(dataset_root).expanduser().resolve()
    opal_dir = (root / "_opal").resolve()
    try:
        opal_dir.relative_to(root)
    except ValueError as exc:
        raise ResponseWindowLabelPromotionError("lineage directory escapes the dataset root.") from exc
    opal_dir.mkdir(parents=True, exist_ok=True)
    return opal_dir


def _confined_manifest_path(value: object, *, root: Path) -> Path:
    if not isinstance(value, str):
        raise ResponseWindowLabelPromotionError("promotion lineage head manifest path must be a string.")
    relative = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or relative.is_absolute()
        or ".." in relative.parts
        or relative.name != PROMOTION_FILENAME
    ):
        raise ResponseWindowLabelPromotionError("promotion lineage head manifest path is not dataset-confined.")
    manifest = (root / Path(*relative.parts)).resolve()
    if not manifest.is_relative_to(root):
        raise ResponseWindowLabelPromotionError("promotion lineage head manifest path escapes the dataset root.")
    return manifest


def _nonnegative_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ResponseWindowLabelPromotionError(f"promotion lineage head {field} is invalid.")
    return value


__all__ = [
    "LINEAGE_HEAD_FILENAME",
    "LineageHead",
    "lineage_publication_lock",
    "load_lineage_head",
    "require_current_parent",
    "update_lineage_head",
]
