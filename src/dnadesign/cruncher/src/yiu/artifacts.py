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
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from dnadesign.cruncher.artifacts.atomic_write import atomic_write_json
from dnadesign.cruncher.utils.hashing import sha256_bytes, sha256_path
from dnadesign.cruncher.yiu.models import YiuStateRecord, YiuValidationReport

ENGINE_CONTRACT_VERSION = "yiu_explicit_v1_1"
STATE_VIEW_SCHEMA_VERSION = 1


def design_id(*, spec_bytes: bytes, catalog_bytes: bytes = b"") -> str:
    return sha256_bytes(spec_bytes + b"\n" + catalog_bytes)[:12]


def input_fingerprint(*, spec_bytes: bytes, catalog_bytes: bytes = b"") -> str:
    return sha256_bytes(spec_bytes + b"\n" + catalog_bytes)


def catalog_fingerprint(*, catalog_bytes: bytes = b"") -> str:
    return sha256_bytes(catalog_bytes)


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


def trace_manifest_path(run_dir: Path) -> Path:
    return run_dir / "yiu_trace_manifest.json"


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


def published_views_manifest_path(run_dir: Path) -> Path:
    return run_dir / "yiu_published_views_manifest.json"


def write_report(run_dir: Path, report: YiuValidationReport) -> Path:
    path = report_path(run_dir)
    atomic_write_json(path, report.model_dump(mode="json"))
    return path


def resolve_code_revision(workspace_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workspace_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, OSError, subprocess.CalledProcessError):
        return None
    revision = result.stdout.strip()
    return revision or None


def write_status(
    run_dir: Path,
    report: YiuValidationReport,
    *,
    input_fingerprint_value: str,
    catalog_fingerprint_value: str,
    code_revision: str | None = None,
) -> Path:
    payload = {
        "stage": "yiu",
        "family": "yiu",
        "protocol": report.protocol,
        "protocol_template": report.protocol_template,
        "status": "completed" if report.status == "satisfied" else "unsatisfied",
        "status_message": f"{report.status} ({report.validation_mode})",
        "spec_name": report.spec_name,
        "state_count": len(report.states),
        "issue_count": len(report.issues),
        "sequence_mode": report.sequence_mode,
        "validation_mode": report.validation_mode,
        "engine_contract_version": ENGINE_CONTRACT_VERSION,
        "view_contract_version": report.metadata.view_contract_version,
        "input_fingerprint": input_fingerprint_value,
        "catalog_fingerprint": catalog_fingerprint_value,
        "code_revision": code_revision,
        "runtime_signature": {
            "cruncher_version": ENGINE_CONTRACT_VERSION,
            "git_sha": code_revision,
            "yiu_contract_version": ENGINE_CONTRACT_VERSION,
            "protocol_template": report.protocol_template or report.protocol,
            "publish_contract_version": report.metadata.view_contract_version or STATE_VIEW_SCHEMA_VERSION,
        },
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
    input_fingerprint_value: str,
    catalog_fingerprint_value: str,
    code_revision: str | None = None,
    catalog_paths: Iterable[Path] = (),
) -> Path:
    artifacts = [
        {"name": "report", "path": report_path(run_dir).name},
        {"name": "status", "path": status_path(run_dir).name},
        {"name": "trace", "path": trace_path(run_dir).name},
        {"name": "trace_manifest", "path": trace_manifest_path(run_dir).name},
        {"name": "parts", "path": parts_path(run_dir).name},
        {"name": "annotations", "path": annotations_path(run_dir).name},
        {"name": "fragments", "path": fragments_path(run_dir).name},
        {"name": "published_views_manifest", "path": published_views_manifest_path(run_dir).name},
        {"name": "published_views", "path": "published/views"},
    ]
    payload = {
        "stage": "yiu",
        "family": "yiu",
        "workflow": "yiu_explicit",
        "protocol": report.protocol,
        "protocol_template": report.protocol_template,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir.resolve()),
        "workspace_root": str(workspace_root.resolve()),
        "spec_name": report.spec_name,
        "status": report.status,
        "state_count": len(report.states),
        "sequence_mode": report.sequence_mode,
        "validation_mode": report.validation_mode,
        "engine_contract_version": ENGINE_CONTRACT_VERSION,
        "view_contract_version": report.metadata.view_contract_version,
        "input_fingerprint": input_fingerprint_value,
        "catalog_fingerprint": catalog_fingerprint_value,
        "code_revision": code_revision,
        "runtime_signature": {
            "cruncher_version": ENGINE_CONTRACT_VERSION,
            "git_sha": code_revision,
            "yiu_contract_version": ENGINE_CONTRACT_VERSION,
            "protocol_template": report.protocol_template or report.protocol,
            "publish_contract_version": report.metadata.view_contract_version or STATE_VIEW_SCHEMA_VERSION,
        },
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


def _state_issue_counts(state: YiuStateRecord, issues: Iterable[Any]) -> tuple[int, int]:
    issue_count = 0
    warning_count = 0
    for issue in issues:
        if issue.state_id != state.state_id and issue.step_id != state.step_id:
            continue
        if issue.severity == "warning":
            warning_count += 1
        else:
            issue_count += 1
    return issue_count, warning_count


def write_trace_manifest(run_dir: Path, report: YiuValidationReport) -> Path:
    states_payload: list[dict[str, Any]] = []
    for state in report.states:
        issue_count, warning_count = _state_issue_counts(state, report.issues)
        states_payload.append(
            {
                "state_id": state.state_id,
                "state_kind": state.state_kind or state.kind,
                "sequence_mode": state.sequence_mode,
                "validation_mode": state.validation_mode,
                "path": f"published/views/{state.state_id}.json",
                "issue_count": issue_count,
                "warning_count": warning_count,
            }
        )
    payload = {
        "schema_version": 1,
        "family": "yiu",
        "protocol": report.protocol,
        "protocol_template": report.protocol_template,
        "spec_name": report.spec_name,
        "state_count": len(report.states),
        "sequence_mode": report.sequence_mode,
        "validation_mode": report.validation_mode,
        "view_contract_version": report.metadata.view_contract_version,
        "states": states_payload,
    }
    path = trace_manifest_path(run_dir)
    atomic_write_json(path, payload)
    return path


def write_published_views_manifest(run_dir: Path, report: YiuValidationReport) -> Path:
    schema_version = report.metadata.view_contract_version or STATE_VIEW_SCHEMA_VERSION
    payload = {
        "schema_version": schema_version,
        "view_contract_version": schema_version,
        "family": "yiu",
        "protocol": report.protocol,
        "protocol_template": report.protocol_template,
        "spec_name": report.spec_name,
        "view_count": len(report.states),
        "sequence_mode": report.sequence_mode,
        "validation_mode": report.validation_mode,
        "views": [
            {
                "state_id": state.state_id,
                "state_kind": state.state_kind or state.kind,
                "path": f"published/views/{state.state_id}.json",
                "sequence_mode": state.sequence_mode,
                "validation_mode": state.validation_mode,
            }
            for state in report.states
        ],
    }
    path = published_views_manifest_path(run_dir)
    atomic_write_json(path, payload)
    return path


def write_csv(path: Path, *, fieldnames: list[str], rows: list[dict[str, Any]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path
