"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/permuter/src/cli/validate.py

CLI wiring for validate Permuter CLI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

from rich.console import Console

from dnadesign.permuter.src.api.validate import validate_dataset
from dnadesign.permuter.src.cli.output import emit_json
from dnadesign.permuter.src.core.storage import append_record_md

console = Console()


def validate(data: Path, strict: bool = False, record: bool = False, as_json: bool = False) -> dict[str, object]:
    report = validate_dataset(data, strict=strict)
    summary: dict[str, object] = {
        "schema": "permuter.validate.v1",
        "ok": report.ok,
        "records": report.records_path,
        "row_count": report.row_count,
        "strict": report.strict,
        "metric_ids": list(report.metric_ids),
        "warnings": list(report.warnings),
    }
    if as_json:
        emit_json(summary)
    else:
        console.print(f"[green]✔[/green] Validation passed for {data}")
    if record:
        try:
            cmd = shlex.join(sys.argv)
        except Exception:
            cmd = " ".join(sys.argv)
        append_record_md(report.records_path.parent, "validate", cmd)
    return summary
