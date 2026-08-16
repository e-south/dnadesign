"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/storage/history_relocation/materialization.py

Materializes a verified OPAL history relocation with path-only projections.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from ...config.types import RootConfig
from ...core.utils import OpalError, file_sha256
from ..locks import CampaignLock
from .contracts import CampaignHistory, HistoryRelocationPlan
from .inspection import (
    canonical_sha256,
    jsonable,
    plan_history_relocation,
    run_artifact_root,
    verified_artifact_path,
)
from .label_ledger import label_ledger_parts, stage_label_ledger
from .run_ledger import stage_canonical_run_ledger
from .state_projection import build_canonical_state

RECEIPT_SCHEMA_VERSION = "opal.history_relocation.v1"


def _file_entry(path: Path, *, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": file_sha256(path),
        "size_bytes": int(path.stat().st_size),
    }


def _files_under(path: Path) -> list[Path]:
    return sorted(candidate for candidate in path.rglob("*") if candidate.is_file())


def _history_entries(history: CampaignHistory) -> list[dict[str, object]]:
    paths: list[Path] = []
    for run in history.runs:
        paths.extend(_files_under(run.round_dir))
        paths.append(run.run_part)
        paths.extend(run.prediction_parts)
    paths.extend(label_ledger_parts(history))
    state_path = history.workdir / "state.json"
    if state_path.is_file():
        paths.append(state_path)
    return sorted(
        (_file_entry(path, root=history.workdir) for path in set(paths)),
        key=lambda item: str(item["path"]),
    )


def _entries_by_path(entries: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for entry in entries:
        path = str(entry["path"])
        existing = result.get(path)
        if existing is not None and existing != entry:
            raise OpalError(f"Campaign history contains conflicting bytes for {path}.")
        result[path] = entry
    return result


def _rebase_value(value: Any, *, source_root: str, target_root: str) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _rebase_value(item, source_root=source_root, target_root=target_root)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_rebase_value(item, source_root=source_root, target_root=target_root) for item in value]
    if isinstance(value, str) and (value == source_root or value.startswith(f"{source_root}/")):
        return f"{target_root}{value[len(source_root) :]}"
    return value


def _rebase_json_file(path: Path, *, source_root: str, target_root: str) -> bool:
    raw = path.read_text(encoding="utf-8")
    if source_root not in raw:
        return False
    if path.suffix == ".jsonl":
        payloads = [json.loads(line) for line in raw.splitlines() if line.strip()]
        rebased = [_rebase_value(payload, source_root=source_root, target_root=target_root) for payload in payloads]
        path.write_text(
            "\n".join(json.dumps(payload, separators=(",", ":"), ensure_ascii=True) for payload in rebased) + "\n",
            encoding="utf-8",
        )
        return True
    payload = json.loads(raw)
    rebased = _rebase_value(payload, source_root=source_root, target_root=target_root)
    path.write_text(json.dumps(rebased, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    return True


def _stage_verified_round(*, run, staging_root: Path) -> Path:
    staged_round = staging_root / "outputs" / "rounds" / f"round_{run.round_index}"
    artifact_root = run_artifact_root(run.round_dir, run_id=run.run_id)
    shutil.copytree(
        artifact_root,
        staged_round / "run_artifacts" / artifact_root.name,
        copy_function=shutil.copy2,
    )
    for artifact_key, receipt in jsonable(run.run_row["artifacts"]).items():
        if receipt is None:
            continue
        if not str(artifact_key).startswith(("metadata/", "model/", "selection/")):
            continue
        source = verified_artifact_path(artifact_root, artifact_key=str(artifact_key))
        target = staged_round / str(artifact_key)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    source_logs = run.round_dir / "logs"
    if source_logs.is_dir():
        shutil.copytree(source_logs, staged_round / "logs", copy_function=shutil.copy2)
    return staged_round


def _stage_history(
    plan: HistoryRelocationPlan, *, cfg: RootConfig, records_path: Path, staging_root: Path
) -> dict[str, Any]:
    source_root = str(plan.source.workdir)
    target_root = str(plan.target.workdir)
    imported_source_entries = _history_entries(plan.source)
    existing_target_entries = _history_entries(plan.target)
    imported_target_entries: list[dict[str, object]] = []
    transformations: list[dict[str, object]] = []
    staged_destinations: list[tuple[Path, Path]] = []
    staged_replacements: list[tuple[Path, Path]] = []
    for run in plan.source.runs:
        staged_round = _stage_verified_round(run=run, staging_root=staging_root)
        for path in _files_under(staged_round / "logs"):
            if path.suffix not in {".json", ".jsonl"}:
                continue
            source_relative = path.relative_to(staging_root)
            source_path = plan.source.workdir / source_relative
            before = file_sha256(source_path)
            if _rebase_json_file(path, source_root=source_root, target_root=target_root):
                transformations.append(
                    {
                        "path": source_relative.as_posix(),
                        "kind": "absolute_path_rebase",
                        "source_sha256": before,
                        "target_sha256": file_sha256(path),
                    }
                )
        staged_destinations.append((staged_round, plan.target.workdir / staged_round.relative_to(staging_root)))
        for prediction_part in run.prediction_parts:
            destination_name = prediction_part.name
            target_prediction = plan.target.workdir / "outputs" / "ledger" / "predictions" / destination_name
            if target_prediction.exists():
                destination_name = f"part-history-r{run.round_index}-{file_sha256(prediction_part)[:16]}.parquet"
            staged_prediction = staging_root / "outputs" / "ledger" / "predictions" / destination_name
            staged_prediction.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(prediction_part, staged_prediction)
            staged_destinations.append(
                (staged_prediction, plan.target.workdir / staged_prediction.relative_to(staging_root))
            )
    staged_runs, run_transformations = stage_canonical_run_ledger(plan, staging_root=staging_root)
    transformations.extend(run_transformations)
    staged_replacements.append((staged_runs, plan.target.workdir / "outputs" / "ledger" / "runs.parquet"))
    staged_labels = stage_label_ledger(plan, staging_root=staging_root)
    if staged_labels is not None:
        if staged_labels[1].exists():
            staged_replacements.append(staged_labels)
        else:
            staged_destinations.append(staged_labels)
    state = build_canonical_state(plan, cfg=cfg, records_path=records_path)
    staged_state = staging_root / "state.json"
    staged_state.write_text(
        json.dumps(state.to_dict(), indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    staged_destinations.append((staged_state, plan.target.workdir / "state.json"))
    for staged_path, target_path in [*staged_destinations, *staged_replacements]:
        if staged_path.is_file() and staged_path != staged_state:
            imported_target_entries.append(_file_entry(staged_path, root=staging_root))
        elif staged_path.is_dir():
            imported_target_entries.extend(_file_entry(path, root=staging_root) for path in _files_under(staged_path))
    imported_target_entries = sorted(imported_target_entries, key=lambda item: str(item["path"]))
    canonical_by_path = _entries_by_path(
        [
            entry
            for entry in existing_target_entries
            if entry["path"] != "state.json"
            and not str(entry["path"]).startswith("outputs/ledger/runs.parquet/")
            and not (staged_labels is not None and str(entry["path"]).startswith("outputs/ledger/labels.parquet/"))
        ]
        + imported_target_entries
        + [_file_entry(staged_state, root=staging_root)]
    )
    canonical_entries = [canonical_by_path[path] for path in sorted(canonical_by_path)]
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "campaign_slug": plan.campaign_slug,
        "imported_rounds": list(plan.imported_rounds),
        "existing_rounds": list(plan.existing_rounds),
        "canonical_rounds": list(plan.canonical_rounds),
        "run_invariant_sha256": plan.invariant_sha256,
        "column_contract_sha256": (None if plan.column_contract is None else plan.column_contract.sha256),
        "imported_source_history_sha256": canonical_sha256(imported_source_entries),
        "existing_target_history_sha256": canonical_sha256(existing_target_entries),
        "canonical_history_sha256": canonical_sha256(canonical_entries),
        "imported_source_files": imported_source_entries,
        "existing_target_files": existing_target_entries,
        "imported_target_files": imported_target_entries,
        "canonical_files": canonical_entries,
        "transformations": sorted(transformations, key=lambda item: str(item["path"])),
    }
    receipt_name = f"history-import-r{plan.imported_rounds[0]}-r{plan.imported_rounds[-1]}.manifest.json"
    staged_receipt = staging_root / "outputs" / "history" / receipt_name
    staged_receipt.parent.mkdir(parents=True, exist_ok=True)
    staged_receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    staged_destinations.append((staged_receipt, plan.target.workdir / staged_receipt.relative_to(staging_root)))
    return {
        "receipt": receipt,
        "receipt_path": plan.target.workdir / staged_receipt.relative_to(staging_root),
        "moves": staged_destinations,
        "replacements": staged_replacements,
    }


def _assert_history_unchanged(history: CampaignHistory, expected: list[dict[str, object]], *, label: str) -> None:
    observed = _history_entries(history)
    if observed != expected:
        raise OpalError(f"{label} campaign history changed while the relocation was staged.")


def _discard_after_commit(path: Path) -> None:
    try:
        shutil.rmtree(path)
    except BaseException:
        pass


def apply_history_relocation(
    plan: HistoryRelocationPlan,
    *,
    cfg: RootConfig,
    records_path: Path,
) -> Path:
    target_parent = plan.target.workdir.parent
    staging_root = Path(tempfile.mkdtemp(prefix=f".{plan.campaign_slug}-history-import-", dir=target_parent))
    created: list[Path] = []
    replaced: list[tuple[Path, Path]] = []
    state_path = plan.target.workdir / "state.json"
    committed = False
    try:
        staged = _stage_history(plan, cfg=cfg, records_path=records_path, staging_root=staging_root)
        imported_source_files = staged["receipt"]["imported_source_files"]
        existing_target_files = staged["receipt"]["existing_target_files"]
        _assert_history_unchanged(plan.source, imported_source_files, label="Source")
        with CampaignLock(plan.target.workdir):
            current = plan_history_relocation(
                source_workdir=plan.source.workdir,
                target_workdir=plan.target.workdir,
                expected_slug=plan.campaign_slug,
                column_contract=plan.column_contract,
            )
            if current.canonical_rounds != plan.canonical_rounds or current.invariant_sha256 != plan.invariant_sha256:
                raise OpalError("Campaign history changed before relocation apply.")
            _assert_history_unchanged(plan.target, existing_target_files, label="Target")
            state_backup = state_path.read_bytes() if state_path.is_file() else None
            state_change_started = False
            try:
                for staged_path, target_path in staged["replacements"]:
                    backup = target_path.with_name(f".{target_path.name}.{staging_root.name}.backup")
                    if backup.exists():
                        raise OpalError(f"History relocation backup destination already exists: {backup}")
                    replaced.append((target_path, backup))
                    os.replace(target_path, backup)
                    os.replace(staged_path, target_path)
                for staged_path, target_path in staged["moves"]:
                    if target_path == state_path:
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        state_change_started = True
                        os.replace(staged_path, target_path)
                        continue
                    if target_path.exists():
                        raise OpalError(f"History relocation destination already exists: {target_path}")
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    created.append(target_path)
                    os.replace(staged_path, target_path)
            except BaseException:
                for path in reversed(created):
                    if path.is_dir():
                        shutil.rmtree(path)
                    elif path.exists():
                        path.unlink()
                if state_change_started:
                    if state_backup is None:
                        state_path.unlink(missing_ok=True)
                    else:
                        state_path.write_bytes(state_backup)
                for target_path, backup in reversed(replaced):
                    if not backup.exists():
                        continue
                    if target_path.is_dir():
                        shutil.rmtree(target_path)
                    elif target_path.exists():
                        target_path.unlink()
                    os.replace(backup, target_path)
                raise
            committed = True
            for _, backup in replaced:
                _discard_after_commit(backup)
        return Path(staged["receipt_path"])
    finally:
        if committed:
            _discard_after_commit(staging_root)
        else:
            shutil.rmtree(staging_root, ignore_errors=True)
