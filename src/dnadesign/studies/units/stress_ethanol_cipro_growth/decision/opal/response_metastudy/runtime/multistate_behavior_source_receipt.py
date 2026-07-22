"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/runtime/multistate_behavior_source_receipt.py

Validate the portable OPAL source receipt embedded in a shadow bundle.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from pathlib import PurePosixPath


def verify_behavior_prediction_source_receipt(source: object) -> dict[str, object]:
    """Require one canonical, run-scoped, portable ledger receipt."""

    if not isinstance(source, dict):
        raise ValueError("behavior prediction source receipt must be a mapping.")
    expected = {
        "run_id",
        "ledger_root",
        "ledger_sha256",
        "files",
        "candidate_count",
        "run_receipt_scored_count",
        "run_lineage",
        "candidate_projection",
    }
    if set(source) != expected:
        raise ValueError("behavior prediction source receipt fields are incomplete or unexpected.")
    if not isinstance(source["run_id"], str) or not source["run_id"].strip():
        raise ValueError("behavior prediction source run_id must be nonempty.")
    if source["ledger_root"] != "outputs/ledger":
        raise ValueError("behavior prediction source ledger root is invalid.")
    for field in ("candidate_count", "run_receipt_scored_count"):
        value = source[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"behavior prediction source {field} must be a positive integer.")
    if source["candidate_count"] != source["run_receipt_scored_count"]:
        raise ValueError("behavior prediction source candidate and run-receipt counts disagree.")
    _verify_run_lineage(source["run_lineage"])
    _verify_candidate_projection(
        source["candidate_projection"],
        candidate_count=int(source["candidate_count"]),
    )
    files = source["files"]
    if not isinstance(files, dict) or set(files) != {"prediction_parts", "run_receipt_parts"}:
        raise ValueError("behavior prediction source file groups are incomplete or unexpected.")
    for group, records in files.items():
        if not isinstance(records, list) or not records:
            raise ValueError(f"behavior prediction source {group} must be a nonempty list.")
        for record in records:
            _verify_file_record(record, group=group)
    observed = (
        "sha256:" + hashlib.sha256(json.dumps(files, separators=(",", ":"), sort_keys=True).encode("utf-8")).hexdigest()
    )
    if source["ledger_sha256"] != observed:
        raise ValueError("behavior prediction source digest disagrees with its file receipts.")
    return source


def _verify_run_lineage(value: object) -> None:
    fields = {
        "as_of_round",
        "model_name",
        "model_params_sha256",
        "y_ingest_name",
        "y_ingest_params_sha256",
        "training_y_ops_sha256",
        "training_row_count",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("behavior prediction run lineage fields are incomplete or unexpected.")
    if value["model_name"] != "random_forest" or value["y_ingest_name"] != "vector_from_table_v1":
        raise ValueError("behavior prediction run model or Y-ingest identity is invalid.")
    for field in ("as_of_round", "training_row_count"):
        item = value[field]
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"behavior prediction run lineage {field} must be a nonnegative integer.")
    if value["training_row_count"] == 0:
        raise ValueError("behavior prediction run lineage must contain training rows.")
    for field in ("model_params_sha256", "y_ingest_params_sha256", "training_y_ops_sha256"):
        _verify_prefixed_digest(value[field], field=field)


def _verify_candidate_projection(value: object, *, candidate_count: int) -> None:
    if not isinstance(value, dict) or set(value) != {"source_row_count", "scored_row_count", "sha256"}:
        raise ValueError("behavior prediction candidate projection receipt is invalid.")
    for field in ("source_row_count", "scored_row_count"):
        item = value[field]
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise ValueError(f"behavior prediction candidate projection {field} must be positive.")
    if value["source_row_count"] < candidate_count or value["scored_row_count"] != candidate_count:
        raise ValueError("behavior prediction candidate projection counts disagree with the scored cohort.")
    _verify_prefixed_digest(value["sha256"], field="candidate_projection.sha256")


def _verify_prefixed_digest(value: object, *, field: str) -> None:
    if (
        not isinstance(value, str)
        or not value.startswith("sha256:")
        or len(value) != 71
        or any(character not in "0123456789abcdef" for character in value.removeprefix("sha256:"))
    ):
        raise ValueError(f"behavior prediction source {field} must be a canonical SHA-256 digest.")


def _verify_file_record(record: object, *, group: str) -> None:
    if not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256"}:
        raise ValueError(f"behavior prediction source {group} file receipt is invalid.")
    raw_path = record["path"]
    if not isinstance(raw_path, str) or "\\" in raw_path:
        raise ValueError("behavior prediction source paths must be portable POSIX paths.")
    path = PurePosixPath(raw_path)
    if path.is_absolute() or ".." in path.parts or path.parts[:2] != ("outputs", "ledger"):
        raise ValueError("behavior prediction source path escapes its logical ledger root.")
    size = record["bytes"]
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("behavior prediction source file size must be a positive integer.")
    digest = record["sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("behavior prediction source file digest must be lowercase SHA-256.")


__all__ = ["verify_behavior_prediction_source_receipt"]
