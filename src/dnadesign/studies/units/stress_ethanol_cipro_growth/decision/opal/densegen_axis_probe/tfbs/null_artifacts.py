"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/densegen_axis_probe/tfbs/null_artifacts.py

Artifact writers for DenseGen TFBS learnability null builds.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from .nulls import TfbsNullBuild


def write_tfbs_null_artifacts(build: TfbsNullBuild, out_dir: Path) -> TfbsNullBuild:
    """Write a null label table and matching viability report with replay hashes."""

    labels_dir = out_dir / "labels"
    manifests_dir = out_dir / "manifests"
    labels_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    report = dict(build.null_viability_report)
    stem = _artifact_stem(report)
    label_table_path = labels_dir / f"{stem}.parquet"
    report_path = manifests_dir / f"{stem}.null_viability_report.json"
    build.labels.to_parquet(label_table_path, index=False, compression="zstd")
    report.update(
        {
            "null_label_table_path": str(label_table_path),
            "null_label_table_hash": _file_sha256(label_table_path),
            "null_label_table_row_count": int(len(build.labels)),
            "null_label_table_schema": list(build.labels.columns),
        }
    )
    _write_json(report_path, report)
    return replace(build, null_viability_report=report)


def _artifact_stem(report: Mapping[str, Any]) -> str:
    null_version = _slug(str(report.get("null_version") or "null"))
    label_name = _slug(str(report.get("label_name") or "label"))
    seed = _slug(str(report.get("seed") or "seed"))
    return f"{null_version}__{label_name}__seed{seed}"


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "value"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
