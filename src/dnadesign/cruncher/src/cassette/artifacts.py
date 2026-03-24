"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cassette/artifacts.py

Artifact paths and persistence helpers for cassette design runs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dnadesign.cruncher.artifacts.atomic_write import atomic_write_json
from dnadesign.cruncher.cassette.models import CassetteEvaluationReport
from dnadesign.cruncher.utils.hashing import sha256_bytes, sha256_path

RUN_META_DIR = "meta"
RUN_PROVENANCE_DIR = "provenance"
RUN_ANALYSIS_DIR = "analysis"
RUN_ANALYSIS_REPORTS_DIR = "reports"
RUN_EXPORT_DIR = "export"


def design_id(*, spec_bytes: bytes, catalog_bytes: bytes) -> str:
    return sha256_bytes(spec_bytes + b"\n" + catalog_bytes)[:12]


def build_run_dir(*, workspace_root: Path, run_root: Path, spec_name: str, cassette_design_id: str) -> Path:
    return workspace_root / run_root / spec_name / cassette_design_id


def ensure_run_dirs(run_dir: Path) -> None:
    (run_dir / RUN_META_DIR).mkdir(parents=True, exist_ok=True)
    (run_dir / RUN_PROVENANCE_DIR).mkdir(parents=True, exist_ok=True)
    (run_dir / RUN_ANALYSIS_DIR / RUN_ANALYSIS_REPORTS_DIR).mkdir(parents=True, exist_ok=True)
    (run_dir / RUN_EXPORT_DIR).mkdir(parents=True, exist_ok=True)


def cassette_manifest_path(run_dir: Path) -> Path:
    return run_dir / RUN_META_DIR / "cassette_manifest.json"


def cassette_status_path(run_dir: Path) -> Path:
    return run_dir / RUN_META_DIR / "cassette_status.json"


def spec_snapshot_path(run_dir: Path) -> Path:
    return run_dir / RUN_PROVENANCE_DIR / "spec_used.yaml"


def catalog_snapshot_path(run_dir: Path) -> Path:
    return run_dir / RUN_PROVENANCE_DIR / "nickase_catalog.yaml"


def report_json_path(run_dir: Path) -> Path:
    return run_dir / RUN_ANALYSIS_DIR / RUN_ANALYSIS_REPORTS_DIR / "report.json"


def report_md_path(run_dir: Path) -> Path:
    return run_dir / RUN_ANALYSIS_DIR / RUN_ANALYSIS_REPORTS_DIR / "report.md"


def render_contract_path(run_dir: Path) -> Path:
    return run_dir / RUN_ANALYSIS_DIR / RUN_ANALYSIS_REPORTS_DIR / "render_contract.json"


def candidate_table_path(run_dir: Path) -> Path:
    return run_dir / RUN_EXPORT_DIR / "table__candidates.csv"


def build_manifest(
    *,
    run_dir: Path,
    workspace_root: Path,
    spec_path: Path,
    catalog_path: Path,
    report: CassetteEvaluationReport,
) -> dict[str, Any]:
    artifacts: list[dict[str, Any]] = [
        {"name": "report_json", "path": str(report_json_path(run_dir).relative_to(run_dir))},
        {"name": "report_md", "path": str(report_md_path(run_dir).relative_to(run_dir))},
        {"name": "candidate_table", "path": str(candidate_table_path(run_dir).relative_to(run_dir))},
        {"name": "spec_snapshot", "path": str(spec_snapshot_path(run_dir).relative_to(run_dir))},
        {"name": "catalog_snapshot", "path": str(catalog_snapshot_path(run_dir).relative_to(run_dir))},
    ]
    if report.render_contract is not None:
        artifacts.append({"name": "render_contract", "path": str(render_contract_path(run_dir).relative_to(run_dir))})
    return {
        "stage": "cassette",
        "workflow": "cassette_design",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir.resolve()),
        "workspace_root": str(workspace_root.resolve()),
        "spec_name": report.spec_name,
        "status": report.status,
        "spec_path": str(spec_path.resolve()),
        "spec_sha256": sha256_path(spec_path),
        "catalog_path": str(catalog_path.resolve()),
        "catalog_sha256": sha256_path(catalog_path),
        "artifacts": artifacts,
    }


def write_manifest(run_dir: Path, manifest: dict[str, Any]) -> Path:
    path = cassette_manifest_path(run_dir)
    atomic_write_json(path, manifest)
    return path


def write_status(run_dir: Path, *, status: str, status_message: str, report: CassetteEvaluationReport) -> Path:
    path = cassette_status_path(run_dir)
    payload = {
        "stage": "cassette",
        "status": status,
        "status_message": status_message,
        "run_dir": str(run_dir.resolve()),
        "spec_name": report.spec_name,
        "issue_count": len(report.issues),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(path, payload)
    return path


def snapshot_inputs(run_dir: Path, *, spec_path: Path, catalog_path: Path) -> None:
    shutil.copyfile(spec_path, spec_snapshot_path(run_dir))
    shutil.copyfile(catalog_path, catalog_snapshot_path(run_dir))


def write_report(run_dir: Path, report: CassetteEvaluationReport, *, markdown: str) -> None:
    atomic_write_json(report_json_path(run_dir), report.model_dump(mode="json"))
    report_md_path(run_dir).write_text(markdown, encoding="utf-8")
    if report.render_contract is not None:
        atomic_write_json(render_contract_path(run_dir), report.render_contract)


def write_candidate_table(run_dir: Path, report: CassetteEvaluationReport) -> None:
    path = candidate_table_path(run_dir)
    fieldnames = [
        "spec_name",
        "status",
        "cassette_sequence",
        "cassette_length",
        "designated_strand",
        "left_nickase",
        "left_nick_coordinate",
        "right_nickase",
        "right_nick_coordinate",
        "bounded_segment_start",
        "bounded_segment_end",
        "bounded_segment_length",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        if report.candidate is None:
            return
        writer.writerow(
            {
                "spec_name": report.spec_name,
                "status": report.status,
                "cassette_sequence": report.candidate.cassette_sequence,
                "cassette_length": report.candidate.cassette_length,
                "designated_strand": report.designated_strand,
                "left_nickase": report.candidate.left_nick.nickase,
                "left_nick_coordinate": report.candidate.left_nick.nick_coordinate,
                "right_nickase": report.candidate.right_nick.nickase,
                "right_nick_coordinate": report.candidate.right_nick.nick_coordinate,
                "bounded_segment_start": report.candidate.bounded_segment.start,
                "bounded_segment_end": report.candidate.bounded_segment.end,
                "bounded_segment_length": report.candidate.bounded_segment.length,
            }
        )


def load_manifest(run_dir: Path) -> dict[str, Any]:
    path = cassette_manifest_path(run_dir)
    if not path.exists():
        raise FileNotFoundError(f"Missing cassette manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_status(run_dir: Path) -> dict[str, Any]:
    path = cassette_status_path(run_dir)
    if not path.exists():
        raise FileNotFoundError(f"Missing cassette status: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
