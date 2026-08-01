"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/runtime/publication.py

Atomic publication and provenance helpers for metastudy bundles.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections.abc import Iterable, Mapping
from pathlib import Path
from uuid import uuid4

_MARIMO_RUNTIME_DIR = "__marimo__"
METASTUDY_SCHEMA_VERSION = "stress_ethanol_cipro_growth.response_metastudy.v13"
HISTORICAL_METASTUDY_SCHEMA_VERSION = "stress_ethanol_cipro_growth.response_metastudy.v12"


def create_staging_dir(final_dir: Path, *, overwrite: bool) -> Path:
    final_dir = final_dir.resolve()
    if final_dir.exists() and any(final_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {final_dir}. Pass --overwrite to publish a replacement bundle."
        )
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = final_dir.parent / f".{final_dir.name}.staging-{uuid4().hex}"
    stage.mkdir()
    return stage


def publish_staging_dir(stage: Path, final_dir: Path, *, overwrite: bool) -> None:
    stage = stage.resolve()
    final_dir = final_dir.resolve()
    if not stage.is_dir() or not any(stage.iterdir()):
        raise RuntimeError(f"Refusing to publish an empty or missing staging bundle: {stage}")
    backup = final_dir.parent / f".{final_dir.name}.backup-{uuid4().hex}"
    moved_existing = False
    try:
        if final_dir.exists():
            if any(final_dir.iterdir()) and not overwrite:
                raise FileExistsError(f"Output directory became non-empty during generation: {final_dir}")
            final_dir.rename(backup)
            moved_existing = True
        stage.rename(final_dir)
    except Exception:
        if moved_existing and backup.exists() and not final_dir.exists():
            backup.rename(final_dir)
        raise
    finally:
        if backup.exists() and final_dir.exists():
            shutil.rmtree(backup)


def remove_staging_dir(stage: Path) -> None:
    if stage.exists():
        shutil.rmtree(stage)


def artifact_inventory(root: Path, artifacts: Mapping[str, Path]) -> dict[str, dict[str, object]]:
    inventory: dict[str, dict[str, object]] = {}
    for artifact_id, path in sorted(artifacts.items()):
        resolved = path.resolve()
        if not resolved.is_relative_to(root.resolve()):
            raise ValueError(f"Artifact escapes bundle root: {resolved}")
        if not resolved.is_file() or resolved.stat().st_size <= 0:
            raise RuntimeError(f"Required artifact is missing or empty: {resolved}")
        inventory[artifact_id] = {
            "path": resolved.relative_to(root.resolve()).as_posix(),
            "bytes": resolved.stat().st_size,
            "sha256": sha256_file(resolved),
        }
    actual = _scientific_artifact_paths(root)
    expected = {path.resolve() for path in artifacts.values()}
    unexpected = sorted(path.relative_to(root.resolve()).as_posix() for path in actual - expected)
    if unexpected:
        raise RuntimeError(f"Bundle contains unregistered artifacts: {unexpected}")
    return inventory


def verify_bundle_artifacts(root: Path) -> dict[str, object]:
    """Verify one published metastudy bundle before interactive review."""

    bundle_root = Path(root).resolve()
    manifest_path = bundle_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Metastudy manifest is missing: {manifest_path}")
    manifest = _load_strict_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("schema_version") not in {
        METASTUDY_SCHEMA_VERSION,
        HISTORICAL_METASTUDY_SCHEMA_VERSION,
    }:
        raise ValueError("Metastudy bundle schema is missing or unsupported.")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("Metastudy manifest has no artifact inventory.")
    expected_paths: set[Path] = set()
    for artifact_id, raw in artifacts.items():
        if not isinstance(artifact_id, str) or not isinstance(raw, dict):
            raise ValueError("Metastudy artifact entries must use string IDs and mappings.")
        relative = raw.get("path")
        size = raw.get("bytes")
        digest = raw.get("sha256")
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"Metastudy artifact {artifact_id!r} lacks a path.")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError(f"Metastudy artifact {artifact_id!r} lacks a positive byte count.")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"Metastudy artifact {artifact_id!r} lacks a sha256 digest.")
        path = (bundle_root / relative).resolve()
        if not path.is_relative_to(bundle_root):
            raise ValueError(f"Metastudy artifact {artifact_id!r} escapes the bundle root.")
        if not path.is_file() or path.stat().st_size != size:
            raise RuntimeError(f"Metastudy artifact {artifact_id!r} is missing or has the wrong size.")
        if sha256_file(path) != digest:
            raise RuntimeError(f"Metastudy artifact {artifact_id!r} digest mismatch.")
        expected_paths.add(path)
    actual_paths = _scientific_artifact_paths(bundle_root) - {manifest_path.resolve()}
    if actual_paths != expected_paths:
        unexpected = sorted(path.relative_to(bundle_root).as_posix() for path in actual_paths - expected_paths)
        missing = sorted(path.relative_to(bundle_root).as_posix() for path in expected_paths - actual_paths)
        raise RuntimeError(f"Metastudy artifact inventory mismatch; unexpected={unexpected}, missing={missing}.")
    return manifest


def source_inventory(repo_root: Path, paths: Iterable[Path]) -> list[dict[str, object]]:
    root = repo_root.resolve()
    rows: list[dict[str, object]] = []
    for path in sorted({candidate.resolve() for candidate in paths}):
        if not path.is_file():
            raise FileNotFoundError(f"Provenance source is missing: {path}")
        rows.append(
            {
                "path": path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def _scientific_artifact_paths(root: Path) -> set[Path]:
    """Return bundle files while excluding Marimo's declared runtime namespace."""

    bundle_root = root.resolve()
    paths: set[Path] = set()
    for path in bundle_root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.resolve().relative_to(bundle_root)
        if relative.parts and relative.parts[0] == _MARIMO_RUNTIME_DIR:
            continue
        paths.add(path.resolve())
    return paths


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_arrays(*arrays: object) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        view = memoryview(array)  # type: ignore[arg-type]
        digest.update(view.cast("B"))
    return digest.hexdigest()


def _load_strict_json(path: Path) -> object:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Metastudy manifest contains duplicate JSON key {key!r}.")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"Metastudy manifest contains non-finite JSON value {value!r}.")

    def finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"Metastudy manifest contains non-finite JSON number {value!r}.")
        return parsed

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
            parse_float=finite_float,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"Metastudy manifest is not valid JSON: {path}") from exc
