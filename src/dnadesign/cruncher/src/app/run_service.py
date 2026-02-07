"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/run_service.py

Enumerate and summarize sample runs in a workspace.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from dnadesign.cruncher.artifacts.entries import normalize_artifacts
from dnadesign.cruncher.artifacts.layout import manifest_path, status_path
from dnadesign.cruncher.config.schema_v3 import CruncherConfig
from dnadesign.cruncher.utils.paths import resolve_run_index_path


@dataclass(frozen=True)
class RunInfo:
    name: str
    stage: str
    created_at: Optional[str]
    run_dir: Path
    status: Optional[str]
    motif_count: int
    pwm_source: Optional[str]
    best_score: float | None
    artifacts: list[dict]
    regulator_set: Optional[dict]
    run_group: Optional[str]

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
            best_score=payload.get("best_score"),
            artifacts=normalize_artifacts(payload.get("artifacts", [])),
            regulator_set=payload.get("regulator_set"),
            run_group=payload.get("run_group"),
        )


@dataclass(frozen=True)
class RunIndexIssue:
    run_name: str
    stage: str
    run_dir: Path
    reason: str


def run_index_path(config_path: Path, catalog_root: Path | str | None = None) -> Path:
    return resolve_run_index_path(config_path)


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


def drop_run_index_entries(
    config_path: Path,
    run_names: Iterable[str],
    *,
    catalog_root: Path | str | None = None,
) -> int:
    index = load_run_index(config_path, catalog_root)
    if not index:
        return 0
    removed = 0
    for name in run_names:
        if name in index:
            del index[name]
            removed += 1
    if removed:
        save_run_index(config_path, index, catalog_root)
    return removed


def find_invalid_run_index_entries(
    config_path: Path,
    *,
    stage: str | None = None,
    catalog_root: Path | str | None = None,
) -> list[RunIndexIssue]:
    index = load_run_index(config_path, catalog_root)
    issues: list[RunIndexIssue] = []
    for run_name, payload in index.items():
        run_stage = str(payload.get("stage") or "unknown")
        if stage is not None and run_stage != stage:
            continue
        run_dir_raw = payload.get("run_dir") or payload.get("run_dir_guess")
        if not run_dir_raw:
            issues.append(
                RunIndexIssue(
                    run_name=run_name,
                    stage=run_stage,
                    run_dir=Path(run_name),
                    reason="missing run_dir",
                )
            )
            continue
        run_dir = Path(str(run_dir_raw)).expanduser()
        if not run_dir.exists():
            issues.append(
                RunIndexIssue(
                    run_name=run_name,
                    stage=run_stage,
                    run_dir=run_dir,
                    reason="missing run directory",
                )
            )
            continue
        if not manifest_path(run_dir).exists():
            issues.append(
                RunIndexIssue(
                    run_name=run_name,
                    stage=run_stage,
                    run_dir=run_dir,
                    reason="missing run_manifest.json",
                )
            )
    return issues


def _iter_run_dirs(stage_dir: Path) -> list[Path]:
    runs: list[Path] = []
    seen: set[Path] = set()
    for manifest_file in sorted(stage_dir.rglob("run_manifest.json")):
        run_dir = manifest_file.parent
        if run_dir.name == "previous":
            continue
        try:
            rel = run_dir.relative_to(stage_dir)
        except ValueError:
            continue
        if rel.parts and rel.parts[0].startswith("."):
            continue
        resolved = run_dir.resolve()
        if resolved in seen:
            continue
        runs.append(run_dir)
        seen.add(resolved)
    return runs


def _merge_payload(existing: dict, updates: dict) -> dict:
    merged = dict(existing)
    for key, value in updates.items():
        if value is None:
            continue
        merged[key] = value
    return merged


def _run_index_key(run_dir: Path, updates: dict) -> str:
    stage = str(updates.get("stage") or "unknown")
    slot = run_dir.name
    run_group = updates.get("run_group")
    if slot in {"latest", "previous"}:
        if isinstance(run_group, str) and run_group:
            return f"{stage}/{run_group}/{slot}"
        parent_name = run_dir.parent.name
        if parent_name and parent_name != stage:
            return f"{stage}/{parent_name}/{slot}"
        return f"{stage}/{slot}"
    return slot


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


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
        "run_group": manifest.get("run_group"),
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
        "run_group": status_payload.get("run_group"),
        "best_score": status_payload.get("best_score"),
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
    run_key = _run_index_key(run_dir, updates)
    return upsert_run_index(config_path, run_key, updates, catalog_root=catalog_root)


def update_run_index_from_status(
    config_path: Path,
    run_dir: Path,
    status_payload: dict,
    *,
    catalog_root: Path | str | None = None,
) -> dict:
    updates = _updates_from_status(status_payload)
    updates.setdefault("run_dir_guess", str(run_dir.resolve()))
    run_key = _run_index_key(run_dir, updates)
    return upsert_run_index(config_path, run_key, updates, catalog_root=catalog_root)


def rebuild_run_index(cfg: CruncherConfig, config_path: Path) -> Path:
    out_dir = config_path.parent / cfg.out_dir
    index: dict[str, dict] = {}
    if out_dir.exists():
        for stage_dir in out_dir.iterdir():
            if not stage_dir.is_dir():
                continue
            for child in _iter_run_dirs(stage_dir):
                manifest_file = manifest_path(child)
                if not manifest_file.exists():
                    continue
                payload = json.loads(manifest_file.read_text())
                entry = _updates_from_manifest(payload)
                status_payload = load_run_status(child)
                if status_payload:
                    entry = _merge_payload(entry, _updates_from_status(status_payload))
                entry.setdefault("run_dir_guess", str(child.resolve()))
                run_key = _run_index_key(child, entry)
                index[run_key] = entry
    return save_run_index(config_path, index, cfg.catalog.catalog_root)


def list_runs(cfg: CruncherConfig, config_path: Path, *, stage: Optional[str] = None) -> list[RunInfo]:
    index = load_run_index(config_path, cfg.catalog.catalog_root)
    runs: list[RunInfo] = []
    if index:
        out_dir = config_path.parent / cfg.out_dir
        for name, payload in index.items():
            if stage and payload.get("stage") != stage:
                continue
            run_dir_raw = payload.get("run_dir") or payload.get("run_dir_guess", name)
            run_dir = Path(run_dir_raw)
            stage_name = str(payload.get("stage") or "")
            if stage_name and not _is_under(run_dir, out_dir / stage_name):
                continue
            runs.append(RunInfo.from_payload(name, payload))
        runs.sort(key=lambda r: r.created_at or "", reverse=True)
        return runs

    out_dir = config_path.parent / cfg.out_dir
    if not out_dir.exists():
        return []
    for stage_dir in out_dir.iterdir():
        if not stage_dir.is_dir():
            continue
        for child in _iter_run_dirs(stage_dir):
            manifest_file = manifest_path(child)
            if not manifest_file.exists():
                continue
            payload = json.loads(manifest_file.read_text())
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
                    best_score=status_payload.get("best_score") if status_payload else None,
                    artifacts=normalize_artifacts(payload.get("artifacts", [])),
                    regulator_set=payload.get("regulator_set"),
                    run_group=payload.get("run_group"),
                )
            )
    runs.sort(key=lambda r: r.created_at or "", reverse=True)
    return runs


def _resolve_run_dir_from_arg(config_path: Path, run_name: str) -> Path | None:
    looks_like_path = Path(run_name).is_absolute() or "/" in run_name or "\\" in run_name
    if not looks_like_path:
        return None
    candidate = Path(run_name).expanduser()
    if candidate.is_absolute():
        resolved = candidate
        if not resolved.exists():
            raise FileNotFoundError(f"Run directory not found: {resolved}")
    else:
        cwd_candidate = (Path.cwd() / candidate).resolve()
        cfg_candidate = (config_path.parent / candidate).resolve()
        cwd_exists = cwd_candidate.exists()
        cfg_exists = cfg_candidate.exists()
        if cwd_exists and cfg_exists and cwd_candidate != cfg_candidate:
            raise FileNotFoundError(
                f"Run path is ambiguous; found both {cwd_candidate} and {cfg_candidate}. Use an absolute path."
            )
        if cwd_exists:
            resolved = cwd_candidate
        elif cfg_exists:
            resolved = cfg_candidate
        else:
            return None
    if resolved.is_file():
        if resolved.name == "run_manifest.json":
            resolved = resolved.parent
        else:
            raise FileNotFoundError(f"Run path points to a file (expected directory): {resolved}")
    if not manifest_path(resolved).exists():
        raise FileNotFoundError(f"Run directory missing manifest: {resolved}")
    return resolved


def get_run(cfg: CruncherConfig, config_path: Path, run_name: str) -> RunInfo:
    run_dir = _resolve_run_dir_from_arg(config_path, run_name)
    if run_dir is not None:
        manifest_file = manifest_path(run_dir)
        payload = json.loads(manifest_file.read_text())
        status_payload = load_run_status(run_dir)
        return RunInfo(
            name=run_dir.name,
            stage=payload.get("stage", "unknown"),
            created_at=payload.get("created_at"),
            run_dir=run_dir,
            status=status_payload.get("status") if status_payload else None,
            motif_count=len(payload.get("motifs", [])),
            pwm_source=(payload.get("motif_store") or {}).get("pwm_source"),
            best_score=status_payload.get("best_score") if status_payload else None,
            artifacts=normalize_artifacts(payload.get("artifacts", [])),
            regulator_set=payload.get("regulator_set"),
            run_group=payload.get("run_group"),
        )
    index = load_run_index(config_path, cfg.catalog.catalog_root)
    if index and run_name in index:
        return RunInfo.from_payload(run_name, index[run_name])
    out_dir = config_path.parent / cfg.out_dir
    run_dir = None
    matches: list[Path] = []
    if out_dir.exists():
        for stage_dir in out_dir.iterdir():
            if not stage_dir.is_dir():
                continue
            for candidate in _iter_run_dirs(stage_dir):
                if not manifest_path(candidate).exists():
                    continue
                candidate_rel = None
                try:
                    candidate_rel = candidate.relative_to(out_dir).as_posix()
                except ValueError:
                    candidate_rel = None
                if candidate.name == run_name or candidate_rel == run_name:
                    matches.append(candidate)
    if len(matches) > 1:
        rendered = ", ".join(str(path) for path in matches[:5])
        raise FileNotFoundError(
            f"Run name '{run_name}' is ambiguous. Matching run directories: {rendered}. "
            "Use an indexed run key or an explicit run path."
        )
    if matches:
        run_dir = matches[0]
    if run_dir is None:
        raise FileNotFoundError(f"Run directory not found (or missing manifest) for run '{run_name}'")
    manifest_file = manifest_path(run_dir)
    payload = json.loads(manifest_file.read_text())
    status_payload = load_run_status(run_dir)
    return RunInfo(
        name=run_name,
        stage=payload.get("stage", "unknown"),
        created_at=payload.get("created_at"),
        run_dir=run_dir,
        status=status_payload.get("status") if status_payload else None,
        motif_count=len(payload.get("motifs", [])),
        pwm_source=(payload.get("motif_store") or {}).get("pwm_source"),
        best_score=status_payload.get("best_score") if status_payload else None,
        artifacts=normalize_artifacts(payload.get("artifacts", [])),
        regulator_set=payload.get("regulator_set"),
        run_group=payload.get("run_group"),
    )


def load_run_status(run_dir: Path) -> Optional[dict]:
    status_file = status_path(run_dir)
    if not status_file.exists():
        return None
    attempts = 3
    delay_s = 0.05
    for attempt in range(attempts):
        try:
            return json.loads(status_file.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            if attempt >= attempts - 1:
                raise ValueError(f"Failed to read run status at {status_file}") from exc
            time.sleep(delay_s * (attempt + 1))
    return None
