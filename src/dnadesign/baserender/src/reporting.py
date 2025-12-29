"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/reporting.py

Run reporting for baserender jobs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Optional


def _hash_mapping(data: Mapping[str, Any]) -> str:
    raw = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return sha256(raw.encode("utf-8")).hexdigest()


@dataclass
class RunReport:
    job_name: str
    input_path: str
    selection_path: Optional[str]
    total_rows_seen: int = 0
    yielded_records: int = 0
    skipped_rows_by_reason: dict[str, int] = field(default_factory=dict)
    skipped_records_by_reason: dict[str, int] = field(default_factory=dict)
    missing_selection_keys: list[str] = field(default_factory=list)
    dropped_by_policy: list[str] = field(default_factory=list)
    outputs: dict[str, str] = field(default_factory=dict)
    style_preset: str = ""
    style_overrides_hash: str = ""

    def note_skip_row(self, reason: str) -> None:
        self.skipped_rows_by_reason[reason] = self.skipped_rows_by_reason.get(reason, 0) + 1

    def note_skip_record(self, reason: str) -> None:
        self.skipped_records_by_reason[reason] = self.skipped_records_by_reason.get(reason, 0) + 1

    def has_skips(self) -> bool:
        return bool(
            self.skipped_rows_by_reason
            or self.skipped_records_by_reason
            or self.missing_selection_keys
            or self.dropped_by_policy
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def write(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())


def style_overrides_hash(overrides: Mapping[str, Any]) -> str:
    return _hash_mapping(overrides)
