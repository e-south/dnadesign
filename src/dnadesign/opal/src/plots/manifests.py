"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/plots/manifests.py

Lightweight manifest contracts for configured OPAL plot artifacts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
import traceback
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..core.utils import OpalError, now_iso, read_json, write_json
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
            outputs.append(_file_entry(data_file, role="tidy_csv"))
    tidy_csv = next((entry["path"] for entry in outputs if entry.get("role") == "tidy_csv"), None)
    warnings: list[dict[str, Any]] = []
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
    return {
        "schema_version": PLOT_ARTIFACT_SCHEMA_VERSION,
        "plot_id": plot_artifact_id(Path(str(context.filename)).stem),
        "name": str(name),
        "kind": str(kind),
        "status": status,
        "started_at": started_at,
        "generated_at": generated_at or now_iso(),
        "run_id": getattr(context, "run_id", None),
        "rounds": _jsonable(getattr(context, "rounds", "unspecified")),
        "params": _jsonable(dict(params)),
        "inputs": [_file_entry(path, role=role) for role, path in sorted(getattr(context, "data_paths", {}).items())],
        "outputs": outputs,
        "tidy_csv": tidy_csv,
        "manifest_path": str(manifest_path),
        "metadata": meta,
        "warnings": warnings,
        "error": _error_entry(error) if error is not None else None,
    }


def write_plot_manifest(manifest: Mapping[str, Any]) -> Path:
    manifest_path = Path(str(manifest["manifest_path"]))
    write_json(manifest_path, _jsonable(dict(manifest)))
    return manifest_path


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


def _file_entry(path: str | Path, *, role: str) -> dict[str, Any]:
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
    return entry


def _error_entry(error: BaseException) -> dict[str, Any]:
    return {
        "category": "PlotDataContractError",
        "type": type(error).__name__,
        "message": str(error),
        "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
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
