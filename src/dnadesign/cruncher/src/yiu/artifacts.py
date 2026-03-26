"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/artifacts.py

Artifact paths and persistence helpers for YIU explicit runs.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from dnadesign.cruncher.artifacts.atomic_write import atomic_write_json
from dnadesign.cruncher.utils.hashing import sha256_bytes, sha256_path
from dnadesign.cruncher.yiu.models import YiuStateRecord, YiuValidationReport


def design_id(*, spec_bytes: bytes, catalog_bytes: bytes = b"") -> str:
    return sha256_bytes(spec_bytes + b"\n" + catalog_bytes)[:12]


def build_run_dir(*, workspace_root: Path, run_root: Path, spec_name: str, run_id: str) -> Path:
    resolved_workspace_root = workspace_root.resolve()
    candidate = resolved_workspace_root.joinpath(run_root, spec_name, run_id).resolve()
    try:
        candidate.relative_to(resolved_workspace_root)
    except ValueError as exc:
        raise ValueError(
            f"YIU run directory must stay inside workspace {resolved_workspace_root}: {candidate}"
        ) from exc
    return candidate


def prepare_run_dir(run_dir: Path, *, force_overwrite: bool) -> None:
    if run_dir.exists():
        if not force_overwrite:
            raise ValueError(f"YIU run directory already exists: {run_dir}. Use --force-overwrite to replace it.")
        shutil.rmtree(run_dir)
    (run_dir / "published" / "views").mkdir(parents=True, exist_ok=True)


def manifest_path(run_dir: Path) -> Path:
    return run_dir / "yiu_manifest.json"


def status_path(run_dir: Path) -> Path:
    return run_dir / "yiu_status.json"


def report_path(run_dir: Path) -> Path:
    return run_dir / "yiu_report.json"


def trace_path(run_dir: Path) -> Path:
    return run_dir / "yiu_trace.jsonl"


def parts_path(run_dir: Path) -> Path:
    return run_dir / "yiu_parts.csv"


def annotations_path(run_dir: Path) -> Path:
    return run_dir / "yiu_annotations.csv"


def fragments_path(run_dir: Path) -> Path:
    return run_dir / "yiu_fragments.csv"


def published_views_dir(run_dir: Path) -> Path:
    return run_dir / "published" / "views"


def state_view_path(run_dir: Path, state_id: str) -> Path:
    return published_views_dir(run_dir) / f"{state_id}.json"


def write_report(run_dir: Path, report: YiuValidationReport) -> Path:
    path = report_path(run_dir)
    atomic_write_json(path, report.model_dump(mode="json"))
    return path


def write_status(run_dir: Path, report: YiuValidationReport) -> Path:
    payload = {
        "stage": "yiu",
        "status": "completed" if report.status == "satisfied" else "unsatisfied",
        "status_message": report.status,
        "spec_name": report.spec_name,
        "issue_count": len(report.issues),
        "run_dir": str(run_dir.resolve()),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    path = status_path(run_dir)
    atomic_write_json(path, payload)
    return path


def write_manifest(
    run_dir: Path,
    *,
    workspace_root: Path,
    spec_path: Path,
    report: YiuValidationReport,
    catalog_paths: Iterable[Path] = (),
) -> Path:
    artifacts = [
        {"name": "report", "path": report_path(run_dir).name},
        {"name": "status", "path": status_path(run_dir).name},
        {"name": "trace", "path": trace_path(run_dir).name},
        {"name": "parts", "path": parts_path(run_dir).name},
        {"name": "annotations", "path": annotations_path(run_dir).name},
        {"name": "fragments", "path": fragments_path(run_dir).name},
        {"name": "published_views", "path": "published/views"},
    ]
    payload = {
        "stage": "yiu",
        "workflow": "yiu_explicit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir.resolve()),
        "workspace_root": str(workspace_root.resolve()),
        "spec_name": report.spec_name,
        "status": report.status,
        "spec_path": str(spec_path.resolve()),
        "spec_sha256": sha256_path(spec_path),
        "catalog_paths": [str(path.resolve()) for path in catalog_paths],
        "artifacts": artifacts,
    }
    path = manifest_path(run_dir)
    atomic_write_json(path, payload)
    return path


def write_trace(run_dir: Path, states: Iterable[YiuStateRecord]) -> Path:
    path = trace_path(run_dir)
    with path.open("w", encoding="utf-8") as handle:
        for state in states:
            handle.write(json.dumps(state.model_dump(mode="json")) + "\n")
    return path


def write_csv(path: Path, *, fieldnames: list[str], rows: list[dict[str, Any]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path
