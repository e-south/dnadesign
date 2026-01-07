"""Run listing helpers and run index management."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dnadesign.cruncher.config.schema_v2 import CruncherConfig
from dnadesign.cruncher.utils.artifacts import normalize_artifacts


@dataclass(frozen=True)
class RunInfo:
    name: str
    stage: str
    created_at: Optional[str]
    run_dir: Path
    status: Optional[str]
    motif_count: int
    pwm_source: Optional[str]
    artifacts: list[dict]
    regulator_set: Optional[dict]

    @classmethod
    def from_payload(cls, name: str, payload: dict) -> "RunInfo":
        run_dir_raw = payload.get("run_dir")
        run_dir = Path(run_dir_raw) if run_dir_raw else Path(payload.get("run_dir_guess", name))
        return cls(
            name=name,
            stage=payload.get("stage", "unknown"),
            created_at=payload.get("created_at"),
            run_dir=run_dir,
            status=payload.get("status"),
            motif_count=int(payload.get("motif_count", 0)),
            pwm_source=payload.get("pwm_source"),
            artifacts=normalize_artifacts(payload.get("artifacts", [])),
            regulator_set=payload.get("regulator_set"),
        )


def run_index_path(config_path: Path, catalog_root: Path | str | None = None) -> Path:
    root = Path(catalog_root) if catalog_root is not None else Path(".cruncher")
    if not root.is_absolute():
        root = config_path.parent / root
    return root / "run_index.json"


def load_run_index(config_path: Path, catalog_root: Path | str | None = None) -> dict[str, dict]:
    path = run_index_path(config_path, catalog_root)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_run_index(config_path: Path, payload: dict[str, dict], catalog_root: Path | str | None = None) -> Path:
    path = run_index_path(config_path, catalog_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def _merge_payload(existing: dict, updates: dict) -> dict:
    merged = dict(existing)
    for key, value in updates.items():
        if value is None:
            continue
        merged[key] = value
    return merged


def upsert_run_index(
    config_path: Path,
    run_name: str,
    updates: dict,
    *,
    catalog_root: Path | str | None = None,
) -> dict:
    index = load_run_index(config_path, catalog_root)
    existing = index.get(run_name, {})
    index[run_name] = _merge_payload(existing, updates)
    save_run_index(config_path, index, catalog_root)
    return index[run_name]


def _updates_from_manifest(manifest: dict) -> dict:
    motifs = manifest.get("motifs") or []
    motif_store = manifest.get("motif_store") or {}
    return {
        "stage": manifest.get("stage", "unknown"),
        "created_at": manifest.get("created_at"),
        "run_dir": manifest.get("run_dir"),
        "config_path": manifest.get("config_path"),
        "config_sha256": manifest.get("config_sha256"),
        "lockfile_sha256": manifest.get("lockfile_sha256"),
        "motif_count": len(motifs),
        "pwm_source": motif_store.get("pwm_source"),
        "regulator_set": manifest.get("regulator_set"),
        "artifacts": manifest.get("artifacts", []),
    }


def _updates_from_status(status_payload: dict) -> dict:
    return {
        "stage": status_payload.get("stage"),
        "status": status_payload.get("status"),
        "created_at": status_payload.get("started_at"),
        "updated_at": status_payload.get("updated_at") or status_payload.get("finished_at"),
        "run_dir": status_payload.get("run_dir"),
        "regulator_set": status_payload.get("regulator_set"),
    }


def _catalog_root_from_manifest(manifest: dict) -> Path | None:
    motif_store = manifest.get("motif_store") or {}
    catalog_root = motif_store.get("catalog_root")
    if catalog_root:
        return Path(catalog_root)
    return None


def update_run_index_from_manifest(
    config_path: Path,
    run_dir: Path,
    manifest: dict,
    *,
    catalog_root: Path | str | None = None,
) -> dict:
    if catalog_root is None:
        catalog_root = _catalog_root_from_manifest(manifest)
    updates = _updates_from_manifest(manifest)
    updates.setdefault("run_dir_guess", str(run_dir.resolve()))
    return upsert_run_index(config_path, run_dir.name, updates, catalog_root=catalog_root)


def update_run_index_from_status(
    config_path: Path,
    run_dir: Path,
    status_payload: dict,
    *,
    catalog_root: Path | str | None = None,
) -> dict:
    updates = _updates_from_status(status_payload)
    updates.setdefault("run_dir_guess", str(run_dir.resolve()))
    return upsert_run_index(config_path, run_dir.name, updates, catalog_root=catalog_root)


def rebuild_run_index(cfg: CruncherConfig, config_path: Path) -> Path:
    out_dir = config_path.parent / cfg.out_dir
    index: dict[str, dict] = {}
    if out_dir.exists():
        for child in out_dir.iterdir():
            if not child.is_dir():
                continue
            manifest_path = child / "run_manifest.json"
            if not manifest_path.exists():
                continue
            payload = json.loads(manifest_path.read_text())
            entry = _updates_from_manifest(payload)
            status_payload = load_run_status(child)
            if status_payload:
                entry = _merge_payload(entry, _updates_from_status(status_payload))
            entry.setdefault("run_dir_guess", str(child.resolve()))
            index[child.name] = entry
    return save_run_index(config_path, index, cfg.motif_store.catalog_root)


def list_runs(cfg: CruncherConfig, config_path: Path, *, stage: Optional[str] = None) -> list[RunInfo]:
    index = load_run_index(config_path, cfg.motif_store.catalog_root)
    runs: list[RunInfo] = []
    if index:
        for name, payload in index.items():
            if stage and payload.get("stage") != stage:
                continue
            runs.append(RunInfo.from_payload(name, payload))
        runs.sort(key=lambda r: r.created_at or "", reverse=True)
        return runs

    out_dir = config_path.parent / cfg.out_dir
    if not out_dir.exists():
        return []
    for child in out_dir.iterdir():
        if not child.is_dir():
            continue
        manifest_path = child / "run_manifest.json"
        if not manifest_path.exists():
            continue
        payload = json.loads(manifest_path.read_text())
        run_stage = payload.get("stage", "unknown")
        if stage and run_stage != stage:
            continue
        status_payload = load_run_status(child)
        runs.append(
            RunInfo(
                name=child.name,
                stage=run_stage,
                created_at=payload.get("created_at"),
                run_dir=child,
                status=status_payload.get("status") if status_payload else None,
                motif_count=len(payload.get("motifs", [])),
                pwm_source=(payload.get("motif_store") or {}).get("pwm_source"),
                artifacts=normalize_artifacts(payload.get("artifacts", [])),
                regulator_set=payload.get("regulator_set"),
            )
        )
    runs.sort(key=lambda r: r.created_at or "", reverse=True)
    return runs


def get_run(cfg: CruncherConfig, config_path: Path, run_name: str) -> RunInfo:
    index = load_run_index(config_path, cfg.motif_store.catalog_root)
    if index and run_name in index:
        return RunInfo.from_payload(run_name, index[run_name])
    out_dir = config_path.parent / cfg.out_dir
    run_dir = out_dir / run_name
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"run_manifest.json not found for run '{run_name}'")
    payload = json.loads(manifest_path.read_text())
    status_payload = load_run_status(run_dir)
    return RunInfo(
        name=run_name,
        stage=payload.get("stage", "unknown"),
        created_at=payload.get("created_at"),
        run_dir=run_dir,
        status=status_payload.get("status") if status_payload else None,
        motif_count=len(payload.get("motifs", [])),
        pwm_source=(payload.get("motif_store") or {}).get("pwm_source"),
        artifacts=normalize_artifacts(payload.get("artifacts", [])),
        regulator_set=payload.get("regulator_set"),
    )


def load_run_status(run_dir: Path) -> Optional[dict]:
    status_path = run_dir / "run_status.json"
    if not status_path.exists():
        return None
    return json.loads(status_path.read_text())
