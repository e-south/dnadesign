"""
Stage-B library artifacts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ...utils.logging_utils import install_native_stderr_filters

LIBRARY_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class LibraryArtifact:
    manifest_path: Path
    builds_path: Path
    members_path: Path
    schema_version: str
    run_id: str
    run_root: str
    config_path: str

    @classmethod
    def load(cls, manifest_path: Path) -> "LibraryArtifact":
        payload = json.loads(manifest_path.read_text())
        return cls(
            manifest_path=manifest_path,
            builds_path=Path(payload.get("library_builds_path", "")),
            members_path=Path(payload.get("library_members_path", "")),
            schema_version=str(payload.get("schema_version")),
            run_id=str(payload.get("run_id")),
            run_root=str(payload.get("run_root")),
            config_path=str(payload.get("config_path")),
        )


def _library_manifest_path(out_dir: Path) -> Path:
    return out_dir / "library_manifest.json"


def write_library_artifact(
    *,
    out_dir: Path,
    builds: list[dict],
    members: list[dict],
    cfg_path: Path,
    run_id: str,
    run_root: Path,
    overwrite: bool = False,
) -> LibraryArtifact:
    out_dir.mkdir(parents=True, exist_ok=True)
    install_native_stderr_filters(suppress_solver_messages=False)
    builds_path = out_dir / "library_builds.parquet"
    members_path = out_dir / "library_members.parquet"

    if not overwrite:
        if builds_path.exists():
            raise FileExistsError(f"Library builds already exist: {builds_path}")
        if members_path.exists():
            raise FileExistsError(f"Library members already exist: {members_path}")

    pd.DataFrame(builds).to_parquet(builds_path, index=False)
    pd.DataFrame(members).to_parquet(members_path, index=False)

    manifest = {
        "schema_version": LIBRARY_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": str(run_id),
        "run_root": str(run_root),
        "config_path": str(cfg_path),
        "library_builds_path": str(builds_path),
        "library_members_path": str(members_path),
    }
    manifest_path = _library_manifest_path(out_dir)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return LibraryArtifact(
        manifest_path=manifest_path,
        builds_path=builds_path,
        members_path=members_path,
        schema_version=LIBRARY_SCHEMA_VERSION,
        run_id=str(run_id),
        run_root=str(run_root),
        config_path=str(cfg_path),
    )


def load_library_artifact(out_dir: Path) -> LibraryArtifact:
    manifest_path = _library_manifest_path(out_dir)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Library manifest not found: {manifest_path}")
    return LibraryArtifact.load(manifest_path)
