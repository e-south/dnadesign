"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/orchestrator/gates.py

Native SGE preflight gates for template QA, submit-shape advice, operator briefing, and log retention.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..runbooks.path_policy import WORKSPACE_RUNTIME_LOGS_RELATIVE_DIR, WORKSPACE_SGE_STDOUT_RELATIVE_DIR
from .usr_overlay_inputs import parse_usr_overlay_guard_inputs

_TEMPLATE_SHEBANG = "#!/bin/bash -l"
_DISALLOWED_NOW_DIRECTIVE = re.compile(r"^\s*#\$\s+-now\s+y(?:\s|$)")
_SLUG_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class SessionCounts:
    running_jobs: int
    queued_jobs: int
    eqw_jobs: int


def _validate_runbook_id(runbook_id: str) -> str:
    text = str(runbook_id).strip()
    if not text:
        raise ValueError("runbook_id must be non-empty")
    if not _SLUG_PATTERN.match(text):
        raise ValueError("runbook_id must be slug-safe ([A-Za-z0-9][A-Za-z0-9._-]*)")
    return text


def ensure_dir_writable(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    if not resolved.exists():
        raise ValueError(f"path does not exist after mkdir: {resolved}")
    if not resolved.is_dir():
        raise ValueError(f"path is not a directory: {resolved}")
    if not os.access(resolved, os.W_OK):
        raise ValueError(f"path is not writable: {resolved}")
    return resolved


LogKind = Literal["sge", "runtime"]


def _validate_stdout_dir_scope(*, stdout_dir: Path, runbook_id: str, log_kind: LogKind) -> Path:
    runbook_id_value = _validate_runbook_id(runbook_id)
    resolved_stdout_dir = stdout_dir.expanduser().resolve()
    if not resolved_stdout_dir.exists():
        raise ValueError(f"stdout_dir does not exist: {resolved_stdout_dir}")
    if not resolved_stdout_dir.is_dir():
        raise ValueError(f"stdout_dir is not a directory: {resolved_stdout_dir}")

    if log_kind == "sge":
        expected_suffix = WORKSPACE_SGE_STDOUT_RELATIVE_DIR / runbook_id_value
        scope_error = f"stdout_dir must be exactly workspace/{WORKSPACE_SGE_STDOUT_RELATIVE_DIR.as_posix()}/<runbook-id>"
    elif log_kind == "runtime":
        expected_suffix = WORKSPACE_RUNTIME_LOGS_RELATIVE_DIR
        scope_error = f"stdout_dir must be exactly workspace/{WORKSPACE_RUNTIME_LOGS_RELATIVE_DIR.as_posix()}"
    else:
        raise ValueError(f"unsupported log kind: {log_kind}")
    suffix_parts = expected_suffix.parts
    resolved_parts = resolved_stdout_dir.parts
    if len(resolved_parts) < len(suffix_parts) or tuple(resolved_parts[-len(suffix_parts) :]) != suffix_parts:
        raise ValueError(scope_error)
    return resolved_stdout_dir


def prune_ops_logs(
    *,
    stdout_dir: Path,
    runbook_id: str,
    log_kind: LogKind = "sge",
    keep_last: int,
    max_age_days: int,
    manifest_path: Path | None = None,
    dry_run: bool = False,
) -> dict[str, object]:
    scoped_stdout_dir = _validate_stdout_dir_scope(
        stdout_dir=stdout_dir,
        runbook_id=runbook_id,
        log_kind=log_kind,
    )
    resolved_manifest_path = (
        (scoped_stdout_dir / "retention-manifest.json")
        if manifest_path is None
        else manifest_path.expanduser().resolve()
    )
    if resolved_manifest_path.parent.resolve() != scoped_stdout_dir:
        raise ValueError("manifest_path must be inside stdout_dir")

    now_epoch = time.time()
    cutoff_epoch = now_epoch - (max_age_days * 86400)
    manifest_name = resolved_manifest_path.name
    candidates = [
        path
        for path in scoped_stdout_dir.iterdir()
        if path.is_file() and path.name != manifest_name
    ]
    candidates.sort(key=lambda path: (path.stat().st_mtime, path.name), reverse=True)
    protected = set(candidates[:keep_last])
    pruned_paths: list[Path] = []
    for path in candidates:
        if path in protected:
            continue
        if path.stat().st_mtime > cutoff_epoch:
            continue
        pruned_paths.append(path)
        if not dry_run:
            path.unlink()

    payload: dict[str, object] = {
        "runbook_id": _validate_runbook_id(runbook_id),
        "log_kind": log_kind,
        "stdout_dir": str(scoped_stdout_dir),
        "manifest_path": str(resolved_manifest_path),
        "keep_last": keep_last,
        "max_age_days": max_age_days,
        "dry_run": bool(dry_run),
        "scanned_count": len(candidates),
        "pruned_count": len(pruned_paths),
        "kept_count": len(candidates) - len(pruned_paths),
        "pruned_files": [str(path) for path in pruned_paths],
    }
    resolved_manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def parse_qstat_output(text: str) -> SessionCounts:
    running = 0
    queued = 0
    eqw = 0
    for line in text.splitlines():
        parts = line.split()
        if not parts or not parts[0].isdigit():
            continue
        state = parts[4] if len(parts) >= 5 else ""
        if "r" in state:
            running += 1
        if "q" in state:
            queued += 1
        if "Eqw" in state:
            eqw += 1
    return SessionCounts(running_jobs=running, queued_jobs=queued, eqw_jobs=eqw)


def qa_submit_template(template: Path) -> list[str]:
    errors: list[str] = []
    if not template.exists():
        return [f"template_missing={template}"]
    if not template.is_file():
        return [f"template_not_file={template}"]
    if not os.access(template, os.R_OK):
        return [f"template_not_readable={template}"]

    text = template.read_text(encoding="utf-8")
    first_line = text.splitlines()[0] if text else ""
    if first_line != _TEMPLATE_SHEBANG:
        errors.append(f"invalid_shebang={template} expected={_TEMPLATE_SHEBANG}")
    if any(_DISALLOWED_NOW_DIRECTIVE.search(line) for line in text.splitlines()):
        errors.append(f"disallowed_queue_bypass_directive={template} directive=-now y")
    return errors


def build_shape_advisor(
    *,
    counts: SessionCounts,
    planned_submits: int,
    warn_over_running: int,
    requires_order: bool,
) -> dict[str, str]:
    if counts.eqw_jobs > 0:
        advisor = "hold"
        reason = "eqw_present"
        recommended_action = "triage_eqw_before_submit"
    elif planned_submits <= 1:
        advisor = "single"
        reason = "single_submit"
        recommended_action = "submit_single"
    elif requires_order:
        advisor = "hold_jid"
        reason = "ordered_pipeline"
        recommended_action = "dependency_chain"
    else:
        advisor = "array"
        reason = "multi_submit"
        recommended_action = "array_or_dependency_chain"

    if counts.running_jobs > warn_over_running and counts.eqw_jobs == 0:
        reason = f"{reason};running_jobs_over_threshold"
        recommended_action = f"confirm_then_{recommended_action}"

    return {
        "advisor": advisor,
        "reason": reason,
        "queue_policy": "respect_queue",
        "recommended_action": recommended_action,
        "running_jobs": str(counts.running_jobs),
        "queued_jobs": str(counts.queued_jobs),
        "eqw_jobs": str(counts.eqw_jobs),
    }


def build_operator_brief(
    *,
    counts: SessionCounts,
    planned_submits: int,
    warn_over_running: int,
) -> tuple[dict[str, str], int]:
    if counts.eqw_jobs > 0:
        submit_gate = "blocked"
        next_action = "triage_eqw"
        exit_code = 2
    elif counts.running_jobs > warn_over_running and planned_submits > 0:
        submit_gate = "confirmation_required"
        next_action = "explicit_confirmation_required"
        exit_code = 0
    else:
        submit_gate = "ready"
        next_action = "submit"
        exit_code = 0

    advisor = "single" if planned_submits <= 1 else "array_or_hold_jid"
    brief = {
        "submit_gate": submit_gate,
        "advisor": advisor,
        "next_action": next_action,
        "queue_policy": "respect_queue",
        "running_jobs": str(counts.running_jobs),
        "queued_jobs": str(counts.queued_jobs),
        "eqw_jobs": str(counts.eqw_jobs),
    }
    return brief, exit_code


def _load_session_counts() -> SessionCounts:
    user = os.environ.get("USER", "")
    result = subprocess.run(
        ["qstat", "-u", user],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip() or "qstat failed"
        raise RuntimeError(stderr)
    return parse_qstat_output(result.stdout)


def _format_record(record: dict[str, str]) -> str:
    return " ".join(f"{key}={value}" for key, value in record.items())


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def _parse_extend_quota(run_args: str) -> int | None:
    text = str(run_args or "").strip()
    if not text:
        return None
    try:
        args = shlex.split(text)
    except ValueError as exc:
        raise ValueError(f"run args must be shell-parseable: {exc}") from exc
    for idx, arg in enumerate(args):
        if arg == "--extend-quota":
            if idx + 1 >= len(args):
                raise ValueError("run args include --extend-quota without a value")
            try:
                value = int(args[idx + 1])
            except ValueError as exc:
                raise ValueError("run args --extend-quota value must be an integer") from exc
            if value <= 0:
                raise ValueError("run args --extend-quota must be > 0")
            return value
        if arg.startswith("--extend-quota="):
            raw_value = arg.split("=", maxsplit=1)[1]
            try:
                value = int(raw_value)
            except ValueError as exc:
                raise ValueError("run args --extend-quota value must be an integer") from exc
            if value <= 0:
                raise ValueError("run args --extend-quota must be > 0")
            return value
    return None


def _load_run_state_total_generated(workspace_root: Path) -> int | None:
    run_state_path = workspace_root / "outputs" / "meta" / "run_state.json"
    if not run_state_path.exists():
        return None
    try:
        payload = json.loads(run_state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid run_state.json: {run_state_path}") from exc
    total_generated_raw = payload.get("total_generated")
    if total_generated_raw is None:
        return None
    try:
        total_generated = int(total_generated_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"run_state total_generated must be an integer: {run_state_path}") from exc
    if total_generated < 0:
        raise ValueError(f"run_state total_generated must be >= 0: {run_state_path}")
    return total_generated


def _planned_rows_for_mode(
    *,
    mode: str,
    run_args: str,
    workspace_root: Path,
    generation_total_quota: int,
) -> int:
    if mode == "fresh":
        return int(generation_total_quota)
    if mode != "resume":
        raise ValueError("mode must be one of: fresh, resume")
    extend_quota = _parse_extend_quota(run_args)
    if extend_quota is not None:
        return int(extend_quota)
    existing_total = _load_run_state_total_generated(workspace_root)
    if existing_total is None:
        return int(generation_total_quota)
    return max(0, int(generation_total_quota) - int(existing_total))


def _project_output_parts(
    *,
    planned_rows: int,
    round_robin: bool,
    max_accepted_per_library: int,
    output_chunk_size: int,
) -> int:
    if planned_rows <= 0:
        return 0
    if round_robin:
        turns = math.ceil(int(planned_rows) / int(max_accepted_per_library))
        parts_per_turn = max(1, math.ceil(int(max_accepted_per_library) / int(output_chunk_size)))
        return int(turns * parts_per_turn)
    return int(math.ceil(int(planned_rows) / int(output_chunk_size)))


def _project_overlay_parts(
    *,
    planned_rows: int,
    round_robin: bool,
    max_accepted_per_library: int,
    usr_chunk_size: int,
) -> int:
    return _project_output_parts(
        planned_rows=planned_rows,
        round_robin=round_robin,
        max_accepted_per_library=max_accepted_per_library,
        output_chunk_size=usr_chunk_size,
    )


def _project_records_parts(
    *,
    planned_rows: int,
    round_robin: bool,
    max_accepted_per_library: int,
    parquet_chunk_size: int,
) -> int:
    return _project_output_parts(
        planned_rows=planned_rows,
        round_robin=round_robin,
        max_accepted_per_library=max_accepted_per_library,
        output_chunk_size=parquet_chunk_size,
    )


def _overlay_parts_paths(*, usr_root: Path, usr_dataset: str, namespace: str) -> tuple[Path, Path]:
    dataset_path = (usr_root / usr_dataset).resolve()
    derived_root = dataset_path / "_derived"
    return derived_root / namespace, derived_root / f"{namespace}.parquet"


def _existing_overlay_parts_count(*, usr_root: Path, usr_dataset: str, namespace: str) -> int:
    parts_dir, parts_file = _overlay_parts_paths(usr_root=usr_root, usr_dataset=usr_dataset, namespace=namespace)
    if parts_dir.exists():
        return len(sorted(parts_dir.glob("part-*.parquet")))
    if parts_file.exists():
        return 1
    return 0


def _compact_overlay_parts(*, usr_root: Path, usr_dataset: str, namespace: str) -> str:
    from dnadesign.usr import Dataset

    dataset = Dataset(usr_root, usr_dataset)
    with dataset.maintenance(reason="ops_overlay_compact"):
        compacted_path = dataset.compact_overlay(namespace)
    return str(compacted_path)


def _records_part_glob(records_path: Path) -> str:
    return f"{records_path.stem}__part-*.parquet"


def _existing_records_part_paths(*, records_path: Path) -> list[Path]:
    return sorted(records_path.parent.glob(_records_part_glob(records_path)))


def _existing_records_parts_count(*, records_path: Path) -> int:
    return len(_existing_records_part_paths(records_path=records_path))


def _oldest_records_part_age_days(*, records_path: Path) -> float:
    part_paths = _existing_records_part_paths(records_path=records_path)
    if not part_paths:
        return 0.0
    oldest_epoch = min(path.stat().st_mtime for path in part_paths)
    age_seconds = max(0.0, time.time() - float(oldest_epoch))
    return float(age_seconds / 86400.0)


def _compact_records_parts(*, records_path: Path) -> str:
    part_paths = _existing_records_part_paths(records_path=records_path)
    if not part_paths:
        return str(records_path)
    try:
        import pyarrow as pa
        import pyarrow.dataset as ds
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ValueError("pyarrow is required for records part compaction") from exc
    sources = [str(path) for path in part_paths]
    if records_path.exists():
        sources.insert(0, str(records_path))
    dataset = ds.dataset(sources, format="parquet")
    tmp_path = records_path.parent / f".{records_path.name}.tmp"
    writer = pq.ParquetWriter(tmp_path, schema=dataset.schema)
    scanner = ds.Scanner.from_dataset(dataset, batch_size=4096)
    try:
        for batch in scanner.to_batches():
            if batch.num_rows == 0:
                continue
            writer.write_table(pa.Table.from_batches([batch], schema=dataset.schema))
    finally:
        writer.close()
    tmp_path.replace(records_path)
    for path in part_paths:
        path.unlink()
    return str(records_path)


def _archived_overlay_root(*, usr_root: Path, usr_dataset: str) -> Path:
    return (usr_root / usr_dataset / "_derived" / "_archived").resolve()


def _archived_overlay_inventory(*, usr_root: Path, usr_dataset: str) -> tuple[int, int]:
    archive_root = _archived_overlay_root(usr_root=usr_root, usr_dataset=usr_dataset)
    if not archive_root.exists():
        return 0, 0
    file_paths = sorted(path for path in archive_root.rglob("*") if path.is_file())
    total_bytes = sum(int(path.stat().st_size) for path in file_paths)
    return len(file_paths), int(total_bytes)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dnadesign.ops.orchestrator.gates")
    subparsers = parser.add_subparsers(dest="command", required=True)

    qa_parser = subparsers.add_parser("qa-submit-preflight", help="Run submit-template QA checks.")
    qa_parser.add_argument("--template", action="append", required=True, help="Template path.")

    shape_parser = subparsers.add_parser("submit-shape-advisor", help="Emit submit-shape guidance.")
    shape_parser.add_argument("--planned-submits", type=_non_negative_int, required=True)
    shape_parser.add_argument("--warn-over-running", type=_positive_int, default=3)
    shape_parser.add_argument("--requires-order", action="store_true")

    brief_parser = subparsers.add_parser("operator-brief", help="Emit submit gate readiness brief.")
    brief_parser.add_argument("--planned-submits", type=_non_negative_int, required=True)
    brief_parser.add_argument("--warn-over-running", type=_positive_int, default=3)

    prune_parser = subparsers.add_parser(
        "prune-ops-logs",
        help="Prune stale ops logs from runbook-scoped stdout directories.",
    )
    prune_parser.add_argument("--stdout-dir", required=True, help="Runbook stdout directory.")
    prune_parser.add_argument(
        "--log-kind",
        choices=("sge", "runtime"),
        default="sge",
        help="Log namespace to prune.",
    )
    prune_parser.add_argument("--runbook-id", required=True, help="Runbook identifier.")
    prune_parser.add_argument("--keep-last", type=_positive_int, required=True)
    prune_parser.add_argument("--max-age-days", type=_positive_int, required=True)
    prune_parser.add_argument("--manifest-path", help="Retention manifest path.")
    prune_parser.add_argument("--dry-run", action="store_true", help="Compute retention without deleting files.")
    prune_parser.add_argument("--json", action="store_true", help="Print JSON payload.")

    ensure_dir_parser = subparsers.add_parser(
        "ensure-dir-writable",
        help="Create a directory and fail-fast if it is not writable.",
    )
    ensure_dir_parser.add_argument("--path", required=True, help="Directory path to create and validate.")

    subparsers.add_parser(
        "session-counts",
        help="Emit running/queued/Eqw qstat counts for the current user.",
    )

    usr_overlay_parser = subparsers.add_parser(
        "usr-overlay-guard",
        help="Fail-fast guard for USR overlay-part growth and optional pre-submit compaction.",
    )
    usr_overlay_parser.add_argument(
        "--tool",
        default="densegen",
        help="Tool config adapter for USR overlay guard inputs.",
    )
    usr_overlay_parser.add_argument("--config", required=True, help="Tool config path.")
    usr_overlay_parser.add_argument("--workspace-root", required=True, help="Tool workspace root path.")
    usr_overlay_parser.add_argument("--mode", choices=("fresh", "resume"), required=True)
    usr_overlay_parser.add_argument("--run-args", required=True, help="DenseGen run args for selected mode.")
    usr_overlay_parser.add_argument(
        "--max-projected-overlay-parts",
        type=_positive_int,
        required=True,
        help="Maximum allowed projected overlay-part count for this run.",
    )
    usr_overlay_parser.add_argument(
        "--max-existing-overlay-parts",
        type=_positive_int,
        required=True,
        help="Maximum allowed existing overlay-part count before submit.",
    )
    usr_overlay_parser.add_argument(
        "--overlay-namespace",
        default="densegen",
        help="Overlay namespace to evaluate for part growth.",
    )
    usr_overlay_parser.add_argument(
        "--auto-compact-existing-overlay-parts",
        action="store_true",
        help="Compact overlay parts in-place when existing count exceeds threshold.",
    )
    usr_overlay_parser.add_argument("--json", action="store_true", help="Print JSON payload.")

    usr_records_part_parser = subparsers.add_parser(
        "usr-records-part-guard",
        help="Fail-fast guard for records__part growth with optional age/count-based pre-submit compaction.",
    )
    usr_records_part_parser.add_argument(
        "--tool",
        default="densegen",
        help="Tool config adapter for records-part guard inputs.",
    )
    usr_records_part_parser.add_argument("--config", required=True, help="Tool config path.")
    usr_records_part_parser.add_argument("--workspace-root", required=True, help="Tool workspace root path.")
    usr_records_part_parser.add_argument("--mode", choices=("fresh", "resume"), required=True)
    usr_records_part_parser.add_argument("--run-args", required=True, help="DenseGen run args for selected mode.")
    usr_records_part_parser.add_argument(
        "--max-projected-records-parts",
        type=_positive_int,
        required=True,
        help="Maximum allowed projected records__part count for this run.",
    )
    usr_records_part_parser.add_argument(
        "--max-existing-records-parts",
        type=_positive_int,
        required=True,
        help="Maximum allowed existing records__part count before submit.",
    )
    usr_records_part_parser.add_argument(
        "--max-existing-records-part-age-days",
        type=_positive_int,
        required=True,
        help="Maximum allowed age in days for the oldest existing records__part file.",
    )
    usr_records_part_parser.add_argument(
        "--auto-compact-existing-records-parts",
        action="store_true",
        help="Compact records__part files in-place when count/age maintenance thresholds are exceeded.",
    )
    usr_records_part_parser.add_argument("--json", action="store_true", help="Print JSON payload.")

    usr_archived_overlay_parser = subparsers.add_parser(
        "usr-archived-overlay-guard",
        help="Fail-fast guard for archived overlay growth under _derived/_archived.",
    )
    usr_archived_overlay_parser.add_argument(
        "--tool",
        default="densegen",
        help="Tool config adapter for archived-overlay guard inputs.",
    )
    usr_archived_overlay_parser.add_argument("--config", required=True, help="Tool config path.")
    usr_archived_overlay_parser.add_argument("--workspace-root", required=True, help="Tool workspace root path.")
    usr_archived_overlay_parser.add_argument(
        "--max-archived-entries",
        type=_non_negative_int,
        required=True,
        help="Maximum allowed archived file count under _derived/_archived before submit.",
    )
    usr_archived_overlay_parser.add_argument(
        "--max-archived-bytes",
        type=_non_negative_int,
        required=True,
        help="Maximum allowed archived bytes under _derived/_archived before submit.",
    )
    usr_archived_overlay_parser.add_argument("--json", action="store_true", help="Print JSON payload.")

    args = parser.parse_args(argv)

    if args.command == "qa-submit-preflight":
        failures: list[str] = []
        for raw_template in args.template:
            template = Path(raw_template).expanduser()
            failures.extend(qa_submit_template(template))
        if failures:
            for failure in failures:
                print(failure, file=sys.stderr)
            return 2
        for raw_template in args.template:
            print(f"qa_preflight=pass template={Path(raw_template).expanduser()}")
        return 0

    if args.command == "submit-shape-advisor":
        counts = _load_session_counts()
        advisor = build_shape_advisor(
            counts=counts,
            planned_submits=args.planned_submits,
            warn_over_running=args.warn_over_running,
            requires_order=bool(args.requires_order),
        )
        print(_format_record(advisor))
        return 0

    if args.command == "operator-brief":
        counts = _load_session_counts()
        brief, exit_code = build_operator_brief(
            counts=counts,
            planned_submits=args.planned_submits,
            warn_over_running=args.warn_over_running,
        )
        print(_format_record(brief))
        return exit_code

    if args.command == "prune-ops-logs":
        manifest_path = Path(args.manifest_path).expanduser() if args.manifest_path else None
        try:
            summary = prune_ops_logs(
                stdout_dir=Path(args.stdout_dir),
                runbook_id=args.runbook_id,
                log_kind=args.log_kind,
                keep_last=args.keep_last,
                max_age_days=args.max_age_days,
                manifest_path=manifest_path,
                dry_run=bool(args.dry_run),
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        except Exception as exc:  # pragma: no cover - defensive guard path
            print(str(exc), file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print(
                _format_record(
                    {
                        "retention_gate": "ok",
                        "runbook_id": str(summary["runbook_id"]),
                        "log_kind": str(summary["log_kind"]),
                        "scanned_count": str(summary["scanned_count"]),
                        "pruned_count": str(summary["pruned_count"]),
                        "kept_count": str(summary["kept_count"]),
                        "manifest_path": str(summary["manifest_path"]),
                    }
                )
            )
        return 0

    if args.command == "ensure-dir-writable":
        try:
            resolved = ensure_dir_writable(Path(args.path))
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(_format_record({"dir_gate": "ok", "path": str(resolved)}))
        return 0

    if args.command == "session-counts":
        counts = _load_session_counts()
        print(
            _format_record(
                {
                    "running_jobs": str(counts.running_jobs),
                    "queued_jobs": str(counts.queued_jobs),
                    "eqw_jobs": str(counts.eqw_jobs),
                }
            )
        )
        return 0

    if args.command == "usr-overlay-guard":
        try:
            inputs = parse_usr_overlay_guard_inputs(tool=args.tool, config_path=Path(args.config))
            workspace_root = Path(args.workspace_root).expanduser().resolve()
            if inputs.run_root is not None and workspace_root != inputs.run_root:
                raise ValueError(
                    "workspace-root must match tool config run root: "
                    f"workspace_root={workspace_root} config_run_root={inputs.run_root}"
                )
            if not inputs.supports_overlay_parts:
                summary = {
                    "guard_status": "skipped",
                    "tool": str(args.tool),
                    "reason": f"tool '{args.tool}' does not emit overlay parts; overlay-part guard not applicable",
                    "config": str(Path(args.config).expanduser().resolve()),
                    "workspace_root": str(workspace_root),
                }
                if args.json:
                    print(json.dumps(summary, sort_keys=True))
                else:
                    print(
                        _format_record(
                            {
                                "guard_status": "skipped",
                                "reason": str(summary["reason"]),
                            }
                        )
                    )
                return 0
            if (
                inputs.round_robin is None
                or inputs.max_accepted_per_library is None
                or inputs.usr_chunk_size is None
                or inputs.generation_total_quota is None
            ):
                raise ValueError("overlay-part guard inputs are incomplete for this producer tool")
            planned_rows = _planned_rows_for_mode(
                mode=args.mode,
                run_args=args.run_args,
                workspace_root=workspace_root,
                generation_total_quota=inputs.generation_total_quota,
            )
            projected_overlay_parts = _project_overlay_parts(
                planned_rows=planned_rows,
                round_robin=inputs.round_robin,
                max_accepted_per_library=inputs.max_accepted_per_library,
                usr_chunk_size=inputs.usr_chunk_size,
            )
            existing_overlay_parts_before = _existing_overlay_parts_count(
                usr_root=inputs.usr_root,
                usr_dataset=inputs.usr_dataset,
                namespace=args.overlay_namespace,
            )
            compacted = False
            compacted_path = ""
            if existing_overlay_parts_before > args.max_existing_overlay_parts:
                if not args.auto_compact_existing_overlay_parts:
                    print(
                        "existing_overlay_parts exceeds threshold; "
                        "run 'uv run usr --root <usr-root> maintenance overlay-compact <dataset> --namespace <namespace>' "
                        "or set --auto-compact-existing-overlay-parts.",
                        file=sys.stderr,
                    )
                    print(
                        _format_record(
                            {
                                "guard_status": "blocked",
                                "overlay_namespace": args.overlay_namespace,
                                "existing_overlay_parts": str(existing_overlay_parts_before),
                                "max_existing_overlay_parts": str(args.max_existing_overlay_parts),
                            }
                        ),
                        file=sys.stderr,
                    )
                    return 2
                compacted_path = _compact_overlay_parts(
                    usr_root=inputs.usr_root,
                    usr_dataset=inputs.usr_dataset,
                    namespace=args.overlay_namespace,
                )
                compacted = True
            existing_overlay_parts_after = _existing_overlay_parts_count(
                usr_root=inputs.usr_root,
                usr_dataset=inputs.usr_dataset,
                namespace=args.overlay_namespace,
            )
            if projected_overlay_parts > args.max_projected_overlay_parts:
                recommended_max_accepted = max(
                    1, math.ceil(int(planned_rows) / int(args.max_projected_overlay_parts))
                )
                print(
                    "projected_overlay_parts exceeds threshold; "
                    "increase runtime.max_accepted_per_library and/or output.usr.chunk_size before submit.",
                    file=sys.stderr,
                )
                print(
                    _format_record(
                        {
                            "guard_status": "blocked",
                            "mode": args.mode,
                            "round_robin": str(inputs.round_robin).lower(),
                            "planned_rows": str(planned_rows),
                            "projected_overlay_parts": str(projected_overlay_parts),
                            "max_projected_overlay_parts": str(args.max_projected_overlay_parts),
                            "max_accepted_per_library": str(inputs.max_accepted_per_library),
                            "usr_chunk_size": str(inputs.usr_chunk_size),
                            "recommended_min_max_accepted_per_library": str(recommended_max_accepted),
                        }
                    ),
                    file=sys.stderr,
                )
                return 2
            summary = {
                "guard_status": "ok",
                "tool": str(args.tool),
                "mode": str(args.mode),
                "round_robin": bool(inputs.round_robin),
                "workspace_root": str(workspace_root),
                "config": str(Path(args.config).expanduser().resolve()),
                "usr_root": str(inputs.usr_root),
                "usr_dataset": str(inputs.usr_dataset),
                "overlay_namespace": str(args.overlay_namespace),
                "planned_rows": int(planned_rows),
                "projected_overlay_parts": int(projected_overlay_parts),
                "max_projected_overlay_parts": int(args.max_projected_overlay_parts),
                "existing_overlay_parts_before": int(existing_overlay_parts_before),
                "existing_overlay_parts_after": int(existing_overlay_parts_after),
                "max_existing_overlay_parts": int(args.max_existing_overlay_parts),
                "compacted": bool(compacted),
                "compacted_path": str(compacted_path),
                "max_accepted_per_library": int(inputs.max_accepted_per_library),
                "usr_chunk_size": int(inputs.usr_chunk_size),
            }
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print(
                _format_record(
                    {
                        "guard_status": "ok",
                        "mode": str(summary["mode"]),
                        "round_robin": str(summary["round_robin"]).lower(),
                        "projected_overlay_parts": str(summary["projected_overlay_parts"]),
                        "existing_overlay_parts_before": str(summary["existing_overlay_parts_before"]),
                        "existing_overlay_parts_after": str(summary["existing_overlay_parts_after"]),
                        "compacted": str(summary["compacted"]).lower(),
                    }
                )
            )
        return 0

    if args.command == "usr-records-part-guard":
        try:
            inputs = parse_usr_overlay_guard_inputs(tool=args.tool, config_path=Path(args.config))
            workspace_root = Path(args.workspace_root).expanduser().resolve()
            if inputs.run_root is not None and workspace_root != inputs.run_root:
                raise ValueError(
                    "workspace-root must match tool config run root: "
                    f"workspace_root={workspace_root} config_run_root={inputs.run_root}"
                )
            if not inputs.supports_records_parts:
                summary = {
                    "guard_status": "skipped",
                    "tool": str(args.tool),
                    "reason": f"tool '{args.tool}' does not emit records part files; records-part guard not applicable",
                    "config": str(Path(args.config).expanduser().resolve()),
                    "workspace_root": str(workspace_root),
                }
                if args.json:
                    print(json.dumps(summary, sort_keys=True))
                else:
                    print(
                        _format_record(
                            {
                                "guard_status": "skipped",
                                "reason": str(summary["reason"]),
                            }
                        )
                    )
                return 0
            if inputs.records_path is None or inputs.parquet_chunk_size is None:
                summary = {
                    "guard_status": "skipped",
                    "tool": str(args.tool),
                    "reason": "records_part_guard requires tool output target 'parquet'",
                    "config": str(Path(args.config).expanduser().resolve()),
                    "workspace_root": str(workspace_root),
                }
                if args.json:
                    print(json.dumps(summary, sort_keys=True))
                else:
                    print(
                        _format_record(
                            {
                                "guard_status": "skipped",
                                "reason": str(summary["reason"]),
                            }
                        )
                    )
                return 0
            if (
                inputs.round_robin is None
                or inputs.max_accepted_per_library is None
                or inputs.generation_total_quota is None
            ):
                raise ValueError("records-part guard inputs are incomplete for this producer tool")
            planned_rows = _planned_rows_for_mode(
                mode=args.mode,
                run_args=args.run_args,
                workspace_root=workspace_root,
                generation_total_quota=inputs.generation_total_quota,
            )
            projected_records_parts = _project_records_parts(
                planned_rows=planned_rows,
                round_robin=inputs.round_robin,
                max_accepted_per_library=inputs.max_accepted_per_library,
                parquet_chunk_size=inputs.parquet_chunk_size,
            )
            records_path = inputs.records_path.resolve()
            existing_records_parts_before = _existing_records_parts_count(records_path=records_path)
            oldest_part_age_days_before = _oldest_records_part_age_days(records_path=records_path)
            compacted = False
            compacted_path = ""
            maintenance_reasons: list[str] = []
            if existing_records_parts_before > args.max_existing_records_parts:
                maintenance_reasons.append("count_threshold")
            if (
                existing_records_parts_before > 0
                and oldest_part_age_days_before > float(args.max_existing_records_part_age_days)
            ):
                maintenance_reasons.append("age_threshold")
            if maintenance_reasons:
                if not args.auto_compact_existing_records_parts:
                    print(
                        "existing records parts require maintenance; "
                        "set --auto-compact-existing-records-parts or compact records__part files before submit.",
                        file=sys.stderr,
                    )
                    print(
                        _format_record(
                            {
                                "guard_status": "blocked",
                                "records_path": str(records_path),
                                "existing_records_parts": str(existing_records_parts_before),
                                "max_existing_records_parts": str(args.max_existing_records_parts),
                                "oldest_existing_records_part_age_days": f"{oldest_part_age_days_before:.2f}",
                                "max_existing_records_part_age_days": str(args.max_existing_records_part_age_days),
                                "maintenance_reasons": ",".join(sorted(set(maintenance_reasons))),
                            }
                        ),
                        file=sys.stderr,
                    )
                    return 2
                compacted_path = _compact_records_parts(records_path=records_path)
                compacted = True
            existing_records_parts_after = _existing_records_parts_count(records_path=records_path)
            oldest_part_age_days_after = _oldest_records_part_age_days(records_path=records_path)
            if projected_records_parts > args.max_projected_records_parts:
                recommended_max_accepted = max(
                    1, math.ceil(int(planned_rows) / int(args.max_projected_records_parts))
                )
                print(
                    "projected_records_parts exceeds threshold; "
                    "increase runtime.max_accepted_per_library and/or output.parquet.chunk_size before submit.",
                    file=sys.stderr,
                )
                print(
                    _format_record(
                        {
                            "guard_status": "blocked",
                            "mode": args.mode,
                            "round_robin": str(inputs.round_robin).lower(),
                            "planned_rows": str(planned_rows),
                            "projected_records_parts": str(projected_records_parts),
                            "max_projected_records_parts": str(args.max_projected_records_parts),
                            "max_accepted_per_library": str(inputs.max_accepted_per_library),
                            "parquet_chunk_size": str(inputs.parquet_chunk_size),
                            "recommended_min_max_accepted_per_library": str(recommended_max_accepted),
                        }
                    ),
                    file=sys.stderr,
                )
                return 2
            summary = {
                "guard_status": "ok",
                "tool": str(args.tool),
                "mode": str(args.mode),
                "round_robin": bool(inputs.round_robin),
                "workspace_root": str(workspace_root),
                "config": str(Path(args.config).expanduser().resolve()),
                "records_path": str(records_path),
                "planned_rows": int(planned_rows),
                "projected_records_parts": int(projected_records_parts),
                "max_projected_records_parts": int(args.max_projected_records_parts),
                "existing_records_parts_before": int(existing_records_parts_before),
                "existing_records_parts_after": int(existing_records_parts_after),
                "max_existing_records_parts": int(args.max_existing_records_parts),
                "oldest_existing_records_part_age_days_before": float(oldest_part_age_days_before),
                "oldest_existing_records_part_age_days_after": float(oldest_part_age_days_after),
                "max_existing_records_part_age_days": int(args.max_existing_records_part_age_days),
                "compacted": bool(compacted),
                "compacted_path": str(compacted_path),
                "max_accepted_per_library": int(inputs.max_accepted_per_library),
                "parquet_chunk_size": int(inputs.parquet_chunk_size),
            }
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print(
                _format_record(
                    {
                        "guard_status": "ok",
                        "mode": str(summary["mode"]),
                        "round_robin": str(summary["round_robin"]).lower(),
                        "projected_records_parts": str(summary["projected_records_parts"]),
                        "existing_records_parts_before": str(summary["existing_records_parts_before"]),
                        "existing_records_parts_after": str(summary["existing_records_parts_after"]),
                        "oldest_existing_records_part_age_days_before": (
                            f"{float(summary['oldest_existing_records_part_age_days_before']):.2f}"
                        ),
                        "compacted": str(summary["compacted"]).lower(),
                    }
                )
            )
        return 0

    if args.command == "usr-archived-overlay-guard":
        try:
            inputs = parse_usr_overlay_guard_inputs(tool=args.tool, config_path=Path(args.config))
            workspace_root = Path(args.workspace_root).expanduser().resolve()
            if inputs.run_root is not None and workspace_root != inputs.run_root:
                raise ValueError(
                    "workspace-root must match tool config run root: "
                    f"workspace_root={workspace_root} config_run_root={inputs.run_root}"
                )
            archived_entries, archived_bytes = _archived_overlay_inventory(
                usr_root=inputs.usr_root,
                usr_dataset=inputs.usr_dataset,
            )
            archive_root = _archived_overlay_root(usr_root=inputs.usr_root, usr_dataset=inputs.usr_dataset)

            blocked_reasons: list[str] = []
            if archived_entries > args.max_archived_entries:
                blocked_reasons.append("archived_entries")
            if archived_bytes > args.max_archived_bytes:
                blocked_reasons.append("archived_bytes")
            if blocked_reasons:
                reason_text = ",".join(blocked_reasons)
                if "archived_entries" in blocked_reasons:
                    print(
                        "archived_entries exceeds threshold; compact or prune _derived/_archived before submit.",
                        file=sys.stderr,
                    )
                if "archived_bytes" in blocked_reasons:
                    print(
                        "archived_bytes exceeds threshold; compact or prune _derived/_archived before submit.",
                        file=sys.stderr,
                    )
                print(
                    _format_record(
                        {
                            "guard_status": "blocked",
                            "reason": reason_text,
                            "tool": str(args.tool),
                            "archive_root": str(archive_root),
                            "archived_entries": str(archived_entries),
                            "max_archived_entries": str(args.max_archived_entries),
                            "archived_bytes": str(archived_bytes),
                            "max_archived_bytes": str(args.max_archived_bytes),
                        }
                    ),
                    file=sys.stderr,
                )
                return 2

            summary = {
                "guard_status": "ok",
                "tool": str(args.tool),
                "workspace_root": str(workspace_root),
                "config": str(Path(args.config).expanduser().resolve()),
                "usr_root": str(inputs.usr_root),
                "usr_dataset": str(inputs.usr_dataset),
                "archive_root": str(archive_root),
                "archived_entries": int(archived_entries),
                "max_archived_entries": int(args.max_archived_entries),
                "archived_bytes": int(archived_bytes),
                "max_archived_bytes": int(args.max_archived_bytes),
            }
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print(
                _format_record(
                    {
                        "guard_status": "ok",
                        "archived_entries": str(summary["archived_entries"]),
                        "archived_bytes": str(summary["archived_bytes"]),
                    }
                )
            )
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
