"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/plots/manifests.py

Lightweight manifest contracts for configured OPAL plot artifacts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import re
import traceback
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..core.utils import ExitCodes, OpalError, file_sha256, now_iso, read_json, write_json
from ..registries.plots import describe_plot_kind

PLOT_ARTIFACT_SCHEMA_VERSION = "opal.plot_artifact.v1"
PLOT_MANIFEST_INDEX_SCHEMA_VERSION = "opal.plot_manifest_index.v1"


def plot_artifact_id(name: str, *, round_suffix: str = "") -> str:
    raw = f"{name}{round_suffix}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._") or "plot"


def plot_manifest_path(output_dir: Path, filename: str) -> Path:
    stem = Path(filename).stem
    return Path(output_dir) / f"{stem}.manifest.json"


def load_plot_artifact_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    payload = read_json(manifest_path)
    if not isinstance(payload, dict):
        raise OpalError(f"Plot artifact manifest is not a JSON object: {manifest_path}")
    if payload.get("schema_version") != PLOT_ARTIFACT_SCHEMA_VERSION:
        raise OpalError(
            f"Unsupported plot artifact manifest schema at {manifest_path}: {payload.get('schema_version')!r}"
        )
    return payload


def load_plot_manifest_index(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    payload = read_json(manifest_path)
    if not isinstance(payload, dict):
        raise OpalError(f"Plot manifest index is not a JSON object: {manifest_path}")
    if payload.get("schema_version") != PLOT_MANIFEST_INDEX_SCHEMA_VERSION:
        raise OpalError(f"Unsupported plot manifest index schema at {manifest_path}: {payload.get('schema_version')!r}")
    return payload


def verified_plot_tidy_csv(manifest: Mapping[str, Any], *, plot_root: str | Path) -> Path:
    """Resolve one manifest-declared tidy table and verify its generation-time digest."""

    tidy_outputs = [
        entry
        for entry in manifest.get("outputs") or []
        if isinstance(entry, Mapping) and entry.get("role") == "tidy_csv"
    ]
    if len(tidy_outputs) != 1:
        raise OpalError(
            f"Plot manifest must declare exactly one tidy_csv output; found {len(tidy_outputs)}.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    declared_text = str(manifest.get("tidy_csv") or "").strip()
    output_text = str(tidy_outputs[0].get("path") or "").strip()
    if not declared_text or not output_text:
        raise OpalError("Plot manifest has an empty tidy_csv path.", ExitCodes.CONTRACT_VIOLATION)
    declared_path = Path(declared_text).expanduser().resolve()
    output_path = Path(output_text).expanduser().resolve()
    if declared_path != output_path:
        raise OpalError(
            "Plot manifest tidy_csv does not match its declared tidy output.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    root = Path(plot_root).expanduser().resolve()
    try:
        declared_path.relative_to(root)
    except ValueError as exc:
        raise OpalError(
            f"Plot tidy CSV is outside the campaign plot root: {declared_path}",
            ExitCodes.CONTRACT_VIOLATION,
        ) from exc
    expected_sha256 = str(tidy_outputs[0].get("sha256") or "").strip().lower()
    if len(expected_sha256) != 64 or any(character not in "0123456789abcdef" for character in expected_sha256):
        raise OpalError(
            "Plot manifest tidy output has no valid SHA-256 digest.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    if not declared_path.is_file():
        raise OpalError(f"Plot tidy CSV is missing: {declared_path}", ExitCodes.CONTRACT_VIOLATION)
    actual_sha256 = file_sha256(declared_path)
    if actual_sha256 != expected_sha256:
        raise OpalError(
            f"Plot tidy CSV SHA-256 does not match its manifest (expected={expected_sha256}, actual={actual_sha256}).",
            ExitCodes.CONTRACT_VIOLATION,
        )
    return declared_path


def build_plot_manifest(
    *,
    name: str,
    kind: str,
    params: Mapping[str, Any],
    context: Any,
    status: str,
    started_at: str,
    generated_at: str | None = None,
    error: BaseException | None = None,
) -> dict[str, Any]:
    output_path = Path(context.output_dir) / str(context.filename)
    manifest_path = plot_manifest_path(Path(context.output_dir), str(context.filename))
    meta = describe_plot_kind(kind)
    outputs = []
    if output_path.exists():
        outputs.append(_file_entry(output_path, role="media"))
    for data_path in getattr(context, "saved_data_paths", []):
        data_file = Path(data_path)
        if data_file.exists():
            outputs.append(_file_entry(data_file, role="tidy_csv", bind_digest=True))
    tidy_csv = next((entry["path"] for entry in outputs if entry.get("role") == "tidy_csv"), None)
    warnings: list[dict[str, Any]] = []
    quality = _quality_entry(tidy_csv=tidy_csv, tidy_schema=meta.get("tidy_schema") or [])
    freshness = _freshness_entry(
        inputs=[_file_entry(path, role=role) for role, path in sorted(getattr(context, "data_paths", {}).items())],
        outputs=outputs,
    )
    if status == "written" and not output_path.exists():
        warnings.append(
            {
                "category": "PlotDataContractError",
                "severity": "warning",
                "message": "Plot plugin returned without writing the expected media output.",
                "path": str(output_path),
            }
        )
        status = "failed"
        error = error or RuntimeError("plot media output was not written")
    if status == "written" and quality.get("tidy_schema_valid") is False:
        missing = ", ".join(quality.get("missing_tidy_columns") or [])
        message = f"Plot tidy CSV is missing declared columns: {missing}"
        warnings.append(
            {
                "category": "PlotDataContractError",
                "severity": "warning",
                "message": message,
                "path": tidy_csv,
            }
        )
        status = "failed"
        error = error or RuntimeError(message)
    caption = _clean_text(params.get("caption")) or meta.get("summary")
    review_purpose = _clean_text(params.get("review_purpose")) or caption
    premise = _clean_text(params.get("premise")) or meta.get("premise")
    decision_value = _clean_text(params.get("decision_value")) or meta.get("decision_value")
    rationale = _clean_text(params.get("rationale")) or meta.get("rationale")
    alt_text = _clean_text(params.get("alt_text")) or meta.get("alt_text")
    non_claim_boundary = _clean_text(params.get("non_claim_boundary")) or meta.get("non_claim_boundary")
    tier = _clean_text(params.get("tier")) or meta.get("tier")
    return {
        "schema_version": PLOT_ARTIFACT_SCHEMA_VERSION,
        "plot_id": plot_artifact_id(Path(str(context.filename)).stem),
        "name": str(name),
        "kind": str(kind),
        "status": status,
        "started_at": started_at,
        "generated_at": generated_at or now_iso(),
        "run_id": getattr(context, "run_id", None),
        "selection_view_id": getattr(context, "selection_view_id", None),
        "rounds": _jsonable(getattr(context, "rounds", "unspecified")),
        "params": _jsonable(dict(params)),
        "artifact_metadata": _jsonable(dict(getattr(context, "artifact_metadata", {}))),
        "inputs": freshness["inputs"],
        "outputs": outputs,
        "tidy_csv": tidy_csv,
        "manifest_path": str(manifest_path),
        "metadata": meta,
        "caption": caption,
        "review_purpose": review_purpose,
        "premise": premise,
        "decision_value": decision_value,
        "rationale": rationale,
        "alt_text": alt_text,
        "non_claim_boundary": non_claim_boundary,
        "tier": tier,
        "quality": quality,
        "freshness": freshness,
        "warnings": warnings,
        "error": _error_entry(error) if error is not None else None,
    }


def write_plot_manifest(manifest: Mapping[str, Any]) -> Path:
    manifest_path = Path(str(manifest["manifest_path"]))
    write_json(manifest_path, _jsonable(dict(manifest)))
    return manifest_path


def refresh_plot_manifest_freshness(manifest: Mapping[str, Any]) -> dict[str, Any]:
    refreshed = dict(manifest)
    inputs = [
        _file_entry(entry["path"], role=str(entry.get("role") or "input"))
        for entry in manifest.get("inputs") or []
        if isinstance(entry, Mapping) and entry.get("path")
    ]
    outputs = [
        _refreshed_file_entry(entry, default_role="output")
        for entry in manifest.get("outputs") or []
        if isinstance(entry, Mapping) and entry.get("path")
    ]
    refreshed["inputs"] = inputs
    refreshed["outputs"] = outputs
    refreshed["freshness"] = _freshness_entry(inputs=inputs, outputs=outputs)
    return refreshed


def write_plot_manifest_index(output_dir: Path, manifests: Iterable[Mapping[str, Any]]) -> Path:
    rows = [dict(row) for row in manifests]
    index = {
        "schema_version": PLOT_MANIFEST_INDEX_SCHEMA_VERSION,
        "generated_at": now_iso(),
        "output_dir": str(Path(output_dir)),
        "plot_count": len(rows),
        "manifests": rows,
    }
    path = Path(output_dir) / "plot_manifest.json"
    write_json(path, _jsonable(index))
    return path


def _file_entry(path: str | Path, *, role: str, bind_digest: bool = False) -> dict[str, Any]:
    file_path = Path(path)
    entry: dict[str, Any] = {
        "role": str(role),
        "path": str(file_path),
        "exists": file_path.exists(),
    }
    if file_path.exists():
        stat = file_path.stat()
        entry.update(
            {
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
        if bind_digest:
            entry["sha256"] = file_sha256(file_path)
    return entry


def _refreshed_file_entry(entry: Mapping[str, Any], *, default_role: str) -> dict[str, Any]:
    refreshed = _file_entry(entry["path"], role=str(entry.get("role") or default_role))
    if entry.get("sha256") is not None:
        refreshed["sha256"] = str(entry["sha256"])
    return refreshed


def _error_entry(error: BaseException) -> dict[str, Any]:
    return {
        "category": "PlotDataContractError",
        "type": type(error).__name__,
        "message": str(error),
        "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
    }


def _quality_entry(*, tidy_csv: str | None, tidy_schema: Iterable[str]) -> dict[str, Any]:
    expected = [str(column) for column in tidy_schema if str(column)]
    quality: dict[str, Any] = {
        "tidy_schema_declared": bool(expected),
        "tidy_schema": expected,
        "tidy_schema_valid": None,
        "missing_tidy_columns": [],
    }
    if tidy_csv is None:
        return quality
    if not expected:
        quality["tidy_schema_valid"] = None
        return quality
    path = Path(tidy_csv)
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            header = next(csv.reader(handle), [])
    except StopIteration:
        header = []
    missing = [column for column in expected if column not in header]
    quality["tidy_schema_valid"] = not missing
    quality["missing_tidy_columns"] = missing
    quality["tidy_columns"] = header
    return quality


def _freshness_entry(*, inputs: list[dict[str, Any]], outputs: list[dict[str, Any]]) -> dict[str, Any]:
    input_mtimes = [
        int(entry["mtime_ns"]) for entry in inputs if entry.get("exists") and isinstance(entry.get("mtime_ns"), int)
    ]
    output_mtimes = [
        int(entry["mtime_ns"]) for entry in outputs if entry.get("exists") and isinstance(entry.get("mtime_ns"), int)
    ]
    latest_input = max(input_mtimes) if input_mtimes else None
    oldest_output = min(output_mtimes) if output_mtimes else None
    if not output_mtimes:
        status = "missing_outputs"
    elif latest_input is not None and oldest_output is not None and oldest_output < latest_input:
        status = "stale"
    else:
        status = "fresh"
    return {
        "schema_version": "opal.plot_freshness.v1",
        "status": status,
        "latest_input_mtime_ns": latest_input,
        "oldest_output_mtime_ns": oldest_output,
        "inputs": inputs,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    return str(value)


def _clean_text(value: Any) -> str:
    text = str(value or "").strip()
    return text
