"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/reporting/artifact_garden.py

Manifest-authoritative inventory and cleanup planning for OPAL artifacts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..analysis.campaign import CampaignAnalysis
from ..core.utils import ExitCodes, OpalError, now_iso
from ..plots.manifests import load_plot_artifact_manifest, load_plot_manifest_index
from .review import load_review_manifest

ARTIFACT_GARDEN_SCHEMA_VERSION = "opal.artifact_garden.v1"
_STALE_EXTENSIONS = {".csv", ".json", ".pdf", ".png", ".svg"}


def build_artifact_garden_audit(config_path: str | Path | None) -> dict[str, Any]:
    """Inventory manifest-backed artifact roots without reading records.parquet."""
    analysis = CampaignAnalysis.from_config_path(Path(config_path) if config_path is not None else None, allow_dir=True)
    cfg = analysis.config
    ws = analysis.workspace
    workdir = ws.workdir.resolve()
    referenced_paths: set[str] = set()
    active_manifests: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    for view in cfg.selection_views:
        review_manifest_path = ws.outputs_dir / "review" / "selection_views" / view.id / "manifest.json"
        if review_manifest_path.exists():
            _collect_review_manifest(
                review_manifest_path,
                referenced_paths=referenced_paths,
                active_manifests=active_manifests,
                warnings=warnings,
            )

    plot_manifest_path = ws.outputs_dir / "plots" / "plot_manifest.json"
    if plot_manifest_path.exists():
        _collect_plot_manifest_index(
            plot_manifest_path,
            referenced_paths=referenced_paths,
            active_manifests=active_manifests,
            warnings=warnings,
        )
    for view in cfg.selection_views:
        view_plot_manifest_path = ws.outputs_dir / "plots" / "selection_views" / view.id / "plot_manifest.json"
        if view_plot_manifest_path.exists():
            _collect_plot_manifest_index(
                view_plot_manifest_path,
                referenced_paths=referenced_paths,
                active_manifests=active_manifests,
                warnings=warnings,
            )

    stale_artifacts: list[dict[str, Any]] = []
    scan_roots = [
        ("review_plots", ws.outputs_dir / "review" / "selection_views"),
        ("configured_plots", ws.outputs_dir / "plots"),
    ]
    for scope, root in scan_roots:
        stale_artifacts.extend(
            _detect_stale_siblings(
                root,
                scope=scope,
                referenced_paths=referenced_paths,
            )
        )

    stale_bytes = sum(int(row.get("size_bytes") or 0) for row in stale_artifacts)
    artifact_roots = [
        _directory_inventory("outputs", ws.outputs_dir),
        _directory_inventory("notebooks", ws.workdir / "notebooks"),
    ]
    bytes_total = sum(int(row.get("size_bytes") or 0) for row in artifact_roots)

    return {
        "schema_version": ARTIFACT_GARDEN_SCHEMA_VERSION,
        "generated_at": now_iso(),
        "root": str(workdir),
        "config_path": str(analysis.config_path),
        "campaign": {
            "name": cfg.campaign.name,
            "slug": cfg.campaign.slug,
        },
        "local_only": _is_local_only_path(workdir),
        "artifact_roots": artifact_roots,
        "active_manifests": active_manifests,
        "stale_artifacts": stale_artifacts,
        "bytes": {
            "artifact_roots": int(bytes_total),
            "stale_artifacts": int(stale_bytes),
        },
        "retention_policy": {
            "mode": "manifest_authoritative",
            "delete_requires_apply": True,
        },
        "prune_plan": {
            "schema_version": "opal.artifact_prune_plan.v1",
            "mode": "stale_artifacts_only",
            "requires_apply": True,
            "item_count": len(stale_artifacts),
            "bytes_to_delete": int(stale_bytes),
            "paths": [
                {
                    "path": row["path"],
                    "scope": row.get("scope"),
                    "size_bytes": int(row.get("size_bytes") or 0),
                    "reason": row.get("reason"),
                }
                for row in stale_artifacts
            ],
        },
        "warnings": warnings,
    }


def prune_stale_artifacts(
    config_path: str | Path | None,
    *,
    apply: bool = False,
) -> dict[str, Any]:
    """Delete stale artifacts only when apply is explicit."""
    audit = build_artifact_garden_audit(config_path)
    if not apply:
        return {
            **audit,
            "applied": False,
            "deleted_count": 0,
            "deleted_paths": [],
            "bytes_deleted": 0,
        }

    workdir = Path(str(audit["root"])).resolve()
    allowed_roots = [
        workdir / "outputs" / "review" / "selection_views",
        workdir / "outputs" / "plots",
    ]
    deleted_paths: list[str] = []
    bytes_deleted = 0
    warnings = list(audit.get("warnings") or [])
    for row in audit.get("stale_artifacts") or []:
        path = Path(str(row.get("path"))).resolve()
        if not _is_relative_to_any(path, allowed_roots):
            raise OpalError(
                f"Refusing to prune artifact outside known generated roots: {path}",
                ExitCodes.CONTRACT_VIOLATION,
            )
        if path.is_symlink():
            warnings.append(
                _warning(
                    "ArtifactGardenWarning",
                    "Skipped stale artifact symlink; remove it manually after inspection.",
                    path=path,
                )
            )
            continue
        if not path.exists():
            continue
        if not path.is_file():
            warnings.append(
                _warning("ArtifactGardenWarning", "Skipped stale artifact because it is not a file.", path=path)
            )
            continue
        size = int(path.stat().st_size)
        path.unlink()
        deleted_paths.append(str(path))
        bytes_deleted += size

    return {
        **audit,
        "warnings": warnings,
        "applied": True,
        "deleted_count": len(deleted_paths),
        "deleted_paths": deleted_paths,
        "bytes_deleted": int(bytes_deleted),
    }


def _collect_review_manifest(
    manifest_path: Path,
    *,
    referenced_paths: set[str],
    active_manifests: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> None:
    referenced_paths.add(_resolved(manifest_path))
    try:
        manifest = load_review_manifest(manifest_path)
    except Exception as exc:
        warnings.append(_warning("ReviewManifestError", str(exc), path=manifest_path))
        return
    active_manifests.append(
        {
            "kind": "review",
            "path": str(manifest_path),
            "schema_version": manifest.get("schema_version"),
            "status": "loaded",
        }
    )
    for row in manifest.get("plots") or []:
        if isinstance(row, dict) and row.get("path"):
            referenced_paths.add(_resolved(row["path"]))
    for path in (manifest.get("artifacts") or {}).values():
        if path:
            referenced_paths.add(_resolved(path))


def _collect_plot_manifest_index(
    index_path: Path,
    *,
    referenced_paths: set[str],
    active_manifests: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> None:
    referenced_paths.add(_resolved(index_path))
    try:
        index = load_plot_manifest_index(index_path)
    except Exception as exc:
        warnings.append(_warning("PlotManifestError", str(exc), path=index_path))
        return
    active_manifests.append(
        {
            "kind": "plot_index",
            "path": str(index_path),
            "schema_version": index.get("schema_version"),
            "status": "loaded",
            "plot_count": int(index.get("plot_count") or len(index.get("manifests") or [])),
        }
    )
    for row in index.get("manifests") or []:
        if not isinstance(row, dict):
            continue
        manifest_path = row.get("manifest_path")
        if manifest_path:
            referenced_paths.add(_resolved(manifest_path))
            path = Path(str(manifest_path))
            if path.exists():
                try:
                    artifact_manifest = load_plot_artifact_manifest(path)
                    active_manifests.append(
                        {
                            "kind": "plot_artifact",
                            "path": str(path),
                            "schema_version": artifact_manifest.get("schema_version"),
                            "status": artifact_manifest.get("status", "loaded"),
                            "plot_id": artifact_manifest.get("plot_id"),
                            "name": artifact_manifest.get("name"),
                        }
                    )
                except Exception as exc:
                    warnings.append(_warning("PlotManifestError", str(exc), path=path))
                    artifact_manifest = row
            else:
                warnings.append(
                    _warning(
                        "PlotManifestError",
                        "Plot manifest index references a missing manifest.",
                        path=path,
                    )
                )
                artifact_manifest = row
        else:
            artifact_manifest = row
        for output in artifact_manifest.get("outputs") or []:
            if isinstance(output, dict) and output.get("path"):
                referenced_paths.add(_resolved(output["path"]))


def _detect_stale_siblings(
    root: Path,
    *,
    scope: str,
    referenced_paths: set[str],
) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    stale = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        if _has_hidden_path_part(path, root=root):
            continue
        if path.suffix.lower() not in _STALE_EXTENSIONS:
            continue
        resolved = _resolved(path)
        if resolved in referenced_paths:
            continue
        stat = path.stat()
        stale.append(
            {
                "category": "ArtifactGardenWarning",
                "severity": "warning",
                "scope": scope,
                "path": str(path),
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "reason": "file is absent from the active OPAL manifest set",
            }
        )
    return stale


def _has_hidden_path_part(path: Path, *, root: Path) -> bool:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        parts = path.parts
    return any(part.startswith(".") for part in parts)


def _directory_inventory(name: str, root: Path) -> dict[str, Any]:
    if not root.exists():
        return {
            "name": name,
            "path": str(root),
            "exists": False,
            "file_count": 0,
            "size_bytes": 0,
        }
    file_count = 0
    size_bytes = 0
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            path = Path(dirpath) / filename
            if not path.is_file() or path.is_symlink():
                continue
            file_count += 1
            size_bytes += int(path.stat().st_size)
    return {
        "name": name,
        "path": str(root),
        "exists": True,
        "file_count": int(file_count),
        "size_bytes": int(size_bytes),
    }


def _is_local_only_path(path: Path) -> bool:
    return ".var" in path.resolve().parts


def _is_relative_to_any(path: Path, roots: list[Path]) -> bool:
    resolved = path.resolve()
    for root in roots:
        try:
            resolved.relative_to(root.resolve())
            return True
        except ValueError:
            continue
    return False


def _resolved(path: str | Path) -> str:
    return str(Path(path).resolve())


def _warning(category: str, message: str, *, path: str | Path | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "category": category,
        "severity": "warning",
        "message": message,
    }
    if path is not None:
        row["path"] = str(path)
    return row
