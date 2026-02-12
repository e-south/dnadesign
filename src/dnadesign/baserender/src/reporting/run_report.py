"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/reporting/run_report.py

Run report model and serialization helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunReport:
    job_name: str
    input_path: str
    selection_path: str | None
    total_rows_seen: int = 0
    yielded_records: int = 0
    skipped_rows_by_reason: dict[str, int] = field(default_factory=dict)
    skipped_records_by_reason: dict[str, int] = field(default_factory=dict)
    missing_selection_keys: list[str] = field(default_factory=list)
    outputs: dict[str, str] = field(default_factory=dict)

    def note_skip_row(self, reason: str) -> None:
        self.skipped_rows_by_reason[reason] = self.skipped_rows_by_reason.get(reason, 0) + 1

    def note_skip_record(self, reason: str) -> None:
        self.skipped_records_by_reason[reason] = self.skipped_records_by_reason.get(reason, 0) + 1

    def has_skips(self) -> bool:
        return bool(self.skipped_rows_by_reason or self.skipped_records_by_reason or self.missing_selection_keys)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())
