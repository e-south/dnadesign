"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/analyze/archive.py

Archive and rewrite analysis outputs for repeated runs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from dnadesign.cruncher.analysis.layout import (
    ANALYSIS_LAYOUT_VERSION,
    PLOT_FILE_PREFIX,
    TABLE_FILE_PREFIX,
    summary_path,
)
from dnadesign.cruncher.artifacts.entries import normalize_artifacts
from dnadesign.cruncher.config.schema_v3 import AnalysisConfig
from dnadesign.cruncher.utils.hashing import sha256_bytes, sha256_path

ANALYSIS_TOP_LEVEL_FILES = {
    "analysis_used.yaml",
    "summary.json",
    "report.json",
    "report.md",
    "plot_manifest.json",
    "table_manifest.json",
    "manifest.json",
    "notebook__run_overview.py",
}
ANALYSIS_TOP_LEVEL_DIRS = {"analysis", "plots", "tables"}


def _is_analysis_relpath(path: str) -> bool:
    if path in ANALYSIS_TOP_LEVEL_FILES:
        return True
    if path.startswith(PLOT_FILE_PREFIX) or path.startswith(TABLE_FILE_PREFIX):
        return True
    for root in ANALYSIS_TOP_LEVEL_DIRS:
        if path == root or path.startswith(f"{root}/"):
            return True
    return False


def _analysis_item_paths(analysis_root: Path) -> list[Path]:
    if not analysis_root.exists():
        return []
    items: list[Path] = []
    for name in sorted(ANALYSIS_TOP_LEVEL_FILES | ANALYSIS_TOP_LEVEL_DIRS):
        path = analysis_root / name
        if path.exists():
            items.append(path)
    for pattern in (f"{PLOT_FILE_PREFIX}*", f"{TABLE_FILE_PREFIX}*"):
        for path in sorted(analysis_root.glob(pattern)):
            if path.exists():
                items.append(path)
    return items


def _prune_latest_analysis_artifacts(manifest: dict) -> None:
    artifacts = normalize_artifacts(manifest.get("artifacts"))
    pruned: list[dict[str, object]] = []
    for item in artifacts:
        if str(item.get("stage") or "") == "analysis":
            continue
        path = str(item.get("path") or "")
        norm_path = path.replace("\\", "/")
        if norm_path == "analysis" or norm_path.startswith("analysis/"):
            if not norm_path.startswith("analysis/_archive/"):
                continue
        if _is_analysis_relpath(norm_path):
            continue
        pruned.append(item)
    manifest["artifacts"] = pruned


def _load_summary_id(analysis_root: Path) -> str | None:
    summary_file = summary_path(analysis_root)
    if not summary_file.exists():
        return None
    try:
        payload = json.loads(summary_file.read_text())
    except Exception as exc:
        raise ValueError(f"analysis summary is not valid JSON: {summary_file}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"analysis summary must be a JSON object: {summary_file}")
    analysis_id = payload.get("analysis_id")
    if not isinstance(analysis_id, str) or not analysis_id:
        raise ValueError(f"analysis summary missing analysis_id: {summary_file}")
    return analysis_id


def _load_summary_payload(analysis_root: Path) -> dict | None:
    summary_file = summary_path(analysis_root)
    if not summary_file.exists():
        return None
    try:
        payload = json.loads(summary_file.read_text())
    except Exception as exc:
        raise ValueError(f"analysis summary is not valid JSON: {summary_file}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"analysis summary must be a JSON object: {summary_file}")
    return payload


def _analysis_signature(
    *,
    analysis_cfg: AnalysisConfig,
    override_payload: dict[str, object] | None,
    config_used_file: Path,
    sequences_file: Path,
    elites_file: Path,
    trace_file: Path,
) -> tuple[str, dict[str, object]]:
    inputs: dict[str, object] = {
        "config_used_sha256": sha256_path(config_used_file),
        "sequences_sha256": sha256_path(sequences_file),
        "elites_sha256": sha256_path(elites_file),
        "analysis_layout_version": ANALYSIS_LAYOUT_VERSION,
    }
    if trace_file.exists():
        inputs["trace_sha256"] = sha256_path(trace_file)
    payload = {
        "analysis": analysis_cfg.model_dump(),
        "analysis_overrides": override_payload or {},
        "inputs": inputs,
    }
    signature = sha256_bytes(json.dumps(payload, sort_keys=True).encode("utf-8"))
    return signature, payload


def _rewrite_manifest_paths(manifest: dict, analysis_id: str, moved_prefixes: list[str]) -> None:
    artifacts = manifest.get("artifacts") or []
    for item in artifacts:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "")
        for prefix in moved_prefixes:
            old_prefix = f"analysis/{prefix}"
            if path == prefix or path.startswith(prefix):
                suffix = path[len(prefix) :]
                item["path"] = f"_archive/{analysis_id}/{prefix}{suffix}"
                break
            if path == old_prefix or path.startswith(old_prefix):
                suffix = path[len(old_prefix) :]
                item["path"] = f"_archive/{analysis_id}/{prefix}{suffix}"
                break


def _update_archived_summary(archive_root: Path, analysis_id: str, moved_prefixes: list[str]) -> None:
    summary_file = summary_path(archive_root)
    if not summary_file.exists():
        return
    payload = json.loads(summary_file.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"analysis summary must be a JSON object: {summary_file}")

    def _rewrite_path(value: str) -> str:
        for prefix in moved_prefixes:
            old_prefix = f"analysis/{prefix}"
            if value == prefix or value.startswith(prefix):
                suffix = value[len(prefix) :]
                return f"_archive/{analysis_id}/{prefix}{suffix}"
            if value == old_prefix or value.startswith(old_prefix):
                suffix = value[len(old_prefix) :]
                return f"_archive/{analysis_id}/{prefix}{suffix}"
        return value

    for key in ("analysis_used", "plot_manifest", "table_manifest"):
        raw = payload.get(key)
        if isinstance(raw, str):
            payload[key] = _rewrite_path(raw)

    artifacts = payload.get("artifacts")
    if isinstance(artifacts, list):
        payload["artifacts"] = [_rewrite_path(item) if isinstance(item, str) else item for item in artifacts]

    payload["analysis_dir"] = str(archive_root.resolve())
    payload["archived_at"] = datetime.now(timezone.utc).isoformat()
    summary_file.write_text(json.dumps(payload, indent=2))


def _archive_existing_analysis(analysis_root: Path, manifest: dict, analysis_id: str) -> None:
    archive_root = analysis_root / "_archive" / analysis_id
    archive_root.mkdir(parents=True, exist_ok=True)
    moved_prefixes: list[str] = []
    for path in _analysis_item_paths(analysis_root):
        if not path.exists():
            continue
        moved_prefixes.append(path.name + ("/" if path.is_dir() else ""))
        shutil.move(str(path), archive_root / path.name)
    if moved_prefixes:
        _rewrite_manifest_paths(manifest, analysis_id, moved_prefixes)
        _update_archived_summary(archive_root, analysis_id, moved_prefixes)


def _clear_latest_analysis(analysis_root: Path) -> None:
    for path in _analysis_item_paths(analysis_root):
        if not path.exists():
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
