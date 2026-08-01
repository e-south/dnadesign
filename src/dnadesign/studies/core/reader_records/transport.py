"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/core/reader_records/transport.py

Bounded JSON transport for Reader's public CLI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from .contracts import READER_CLI_SCHEMA, READER_RECORD_SCHEMA_VERSION, ReaderRecordExpectation
from .validation import ReaderDataframeRecordError, ReaderRecordError, list_value, mapping, text

PAGE_LIMIT = 100
MAX_RECORD_PAGES = 100
READER_CLI_TIMEOUT_SECONDS = 60


def collect_record_pages(
    command: Sequence[str], *, config_path: Path, cwd: Path
) -> tuple[Mapping[str, object], tuple[Mapping[str, object], ...]]:
    """Collect a bounded, internally stable sequence of record pages."""

    continuation: str | None = None
    first_data: Mapping[str, object] | None = None
    records: list[Mapping[str, object]] = []
    seen_ids: set[str] = set()
    seen_continuations: set[str] = set()
    page_count = 0
    while True:
        page_count += 1
        argv = [*command, "records", str(config_path), "--limit", str(PAGE_LIMIT), "--format", "json"]
        if continuation is not None:
            argv.extend(("--continuation", continuation))
        payload = run_reader_json(argv, cwd=cwd)
        if (
            payload.get("schema") != READER_CLI_SCHEMA
            or payload.get("command") != "records"
            or payload.get("ok") is not True
        ):
            raise ReaderDataframeRecordError(
                "Reader records command did not return a successful reader.cli/v1 records payload"
            )
        data = mapping(payload.get("data"), label="records.data")
        page_records = list_value(data.get("records"), label="records.data.records")
        if first_data is None:
            first_data = data
        else:
            for field in ("experiment", "catalog", "selection", "summary"):
                if data.get(field) != first_data.get(field):
                    raise ReaderDataframeRecordError(f"Reader records pagination changed data.{field}")
        for index, value in enumerate(page_records):
            record = mapping(value, label=f"records.data.records[{index}]")
            current_id = text(record.get("record_id"), label=f"records.data.records[{index}].record_id")
            if current_id in seen_ids:
                raise ReaderDataframeRecordError(f"Reader records pagination repeated record_id {current_id!r}")
            seen_ids.add(current_id)
            records.append(record)
        meta = mapping(payload.get("meta"), label="records.meta")
        truncated = meta.get("truncated")
        if not isinstance(truncated, bool):
            raise ReaderDataframeRecordError("records.meta.truncated must be a boolean")
        next_token = meta.get("continuation")
        if not truncated:
            if next_token is not None:
                raise ReaderDataframeRecordError("records.meta.continuation must be null on the final page")
            break
        if not page_records:
            raise ReaderDataframeRecordError("Reader records truncated page must contain at least one record")
        next_continuation = text(next_token, label="records.meta.continuation")
        if next_continuation in seen_continuations:
            raise ReaderDataframeRecordError(
                f"Reader records pagination repeated continuation token {next_continuation!r}"
            )
        if page_count >= MAX_RECORD_PAGES:
            raise ReaderDataframeRecordError(
                f"Reader records pagination exceeded the {MAX_RECORD_PAGES}-page safety bound"
            )
        seen_continuations.add(next_continuation)
        continuation = next_continuation
    assert first_data is not None
    summary = mapping(first_data.get("summary"), label="records.data.summary")
    if summary.get("records") != len(records):
        raise ReaderDataframeRecordError(
            f"Reader records summary count {summary.get('records')!r} does not match collected count {len(records)}"
        )
    return first_data, tuple(records)


def verify_record_store(
    command: Sequence[str],
    *,
    config_path: Path,
    cwd: Path,
    expected_records: Mapping[str, ReaderRecordExpectation],
) -> None:
    """Require Reader's full provenance verifier inside the stable catalog read."""

    payload = run_reader_json([*command, "verify", str(config_path), "--format", "json"], cwd=cwd)
    if (
        payload.get("schema") != READER_CLI_SCHEMA
        or payload.get("command") != "verify"
        or payload.get("ok") is not True
    ):
        raise ReaderRecordError("Reader verify did not return a successful reader.cli/v1 verify payload")
    report = mapping(payload.get("data"), label="verify.data")
    if set(report) != {"schema", "status", "summary", "issues", "records"}:
        raise ReaderRecordError("Reader verify report fields are malformed")
    if report.get("schema") != "reader.verify/v1" or report.get("status") != "ok":
        raise ReaderRecordError("Reader verify status must be ok before records can be consumed")
    if list_value(report.get("issues"), label="verify.data.issues"):
        raise ReaderRecordError("Reader verify reported issues despite status ok")
    summary = mapping(report.get("summary"), label="verify.data.summary")
    rows = list_value(report.get("records"), label="verify.data.records")
    summary_fields = {
        "checked",
        "failed",
        "unverifiable",
        "invocations_checked",
        "invocation_failures",
    }
    if set(summary) != summary_fields or any(type(summary.get(field)) is not int for field in summary_fields):
        raise ReaderRecordError("Reader verify summary is malformed")
    if (
        summary.get("checked") != len(rows)
        or summary.get("failed") != 0
        or summary.get("unverifiable") != 0
        or summary.get("invocation_failures") != 0
        or summary["invocations_checked"] < 1
    ):
        raise ReaderRecordError("Reader verify summary is not a complete successful provenance check")
    verified_by_id: dict[str, Mapping[str, object]] = {}
    for index, value in enumerate(rows):
        row = mapping(value, label=f"verify.data.records[{index}]")
        if set(row) != {"record_id", "kind", "schema_version", "status", "issues"}:
            raise ReaderRecordError(f"Reader verify record row {index} fields are malformed")
        record_id = text(row.get("record_id"), label=f"verify.data.records[{index}].record_id")
        row_issues = list_value(row.get("issues"), label=f"verify.data.records[{index}].issues")
        if row.get("status") != "ok" or row_issues:
            raise ReaderRecordError(f"Reader verify record {record_id!r} is not cleanly verified")
        if record_id in verified_by_id:
            raise ReaderRecordError(f"Reader verify repeated record_id {record_id!r}")
        verified_by_id[record_id] = row
    for expectation in expected_records.values():
        row = verified_by_id.get(expectation.record_id)
        if (
            row is None
            or row.get("kind") != expectation.kind
            or row.get("schema_version") != READER_RECORD_SCHEMA_VERSION
        ):
            raise ReaderRecordError(
                f"Reader verify did not confirm expected record {expectation.record_id!r} as schema-v6 status ok"
            )


def reader_command(reader_root: Path) -> tuple[str, ...]:
    """Select the canonical Reader CLI invocation for a repository."""

    repository_executable = reader_root / ".venv" / "bin" / "reader"
    if repository_executable.is_file():
        return (str(repository_executable),)
    installed = shutil.which("reader")
    if installed:
        return (installed,)
    uv = shutil.which("uv")
    if uv:
        return (uv, "run", "--project", str(reader_root), "reader")
    raise ReaderDataframeRecordError("Reader public CLI is unavailable")


def run_reader_json(command: Sequence[str], *, cwd: Path) -> Mapping[str, object]:
    """Run one bounded Reader CLI command and validate its JSON envelope."""

    environment = os.environ.copy()
    environment.pop("__PYVENV_LAUNCHER__", None)
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            timeout=READER_CLI_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise ReaderDataframeRecordError(
            f"Reader CLI command timed out after {READER_CLI_TIMEOUT_SECONDS} seconds"
        ) from exc
    raw = completed.stdout.strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ReaderDataframeRecordError(
            f"Reader records command returned invalid JSON: {raw or completed.stderr.strip() or '<empty>'}"
        ) from exc
    result = mapping(payload, label="Reader CLI envelope")
    if completed.returncode != 0:
        error = result.get("error")
        raise ReaderDataframeRecordError(f"Reader records command failed: {json.dumps(error, sort_keys=True)}")
    return result
