"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/reporting/run_report.py

Run report model and serialization helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..io.captured_source import CapturedSource


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    output_metrics: dict[str, dict[str, int | float | str]] = field(default_factory=dict)
    _source_evidence: dict[str, CapturedSource] = field(default_factory=dict, init=False, repr=False)

    def note_skip_row(self, reason: str) -> None:
        self.skipped_rows_by_reason[reason] = self.skipped_rows_by_reason.get(reason, 0) + 1

    def note_skip_record(self, reason: str) -> None:
        self.skipped_records_by_reason[reason] = self.skipped_records_by_reason.get(reason, 0) + 1

    def has_skips(self) -> bool:
        return bool(self.skipped_rows_by_reason or self.skipped_records_by_reason or self.missing_selection_keys)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_name": self.job_name,
            "input_path": self.input_path,
            "selection_path": self.selection_path,
            "total_rows_seen": self.total_rows_seen,
            "yielded_records": self.yielded_records,
            "skipped_rows_by_reason": dict(self.skipped_rows_by_reason),
            "skipped_records_by_reason": dict(self.skipped_records_by_reason),
            "missing_selection_keys": list(self.missing_selection_keys),
            "outputs": dict(self.outputs),
            "output_metrics": {key: dict(value) for key, value in self.output_metrics.items()},
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())

    def capture_source_evidence(self) -> None:
        sources = {"input": self.input_path}
        if self.selection_path is not None:
            sources["selection"] = self.selection_path
        self._source_evidence = {label: CapturedSource.capture(source) for label, source in sources.items()}

    def source_content(self, label: str) -> bytes:
        evidence = self._source_evidence.get(label)
        if evidence is None:
            raise ValueError(f"Render source evidence was not captured for {label}")
        if evidence.content is None:
            raise ValueError(f"Render source bytes were already released for {label}")
        return evidence.content

    def release_source_content(self) -> None:
        self._source_evidence = {label: evidence.without_content() for label, evidence in self._source_evidence.items()}

    def verify_source_evidence(self) -> None:
        if not self._source_evidence:
            raise ValueError("Render source evidence was not captured before execution")
        sources = {"input": self.input_path}
        if self.selection_path is not None:
            sources["selection"] = self.selection_path
        for label, source in sources.items():
            expected = self._source_evidence.get(label)
            if expected is None:
                raise ValueError(f"Render source changed during execution: {source}")
            expected.verify_unchanged()

    def write_portable_manifest(self, path: Path, *, bundle_root: Path, staging_root: Path) -> None:
        """Write the portable, complete catalog for one immutable render bundle."""
        final_root = bundle_root.resolve(strict=False)

        if not self._source_evidence:
            self.capture_source_evidence()

        portable_outputs: dict[str, str] = {}
        for key, value in self.outputs.items():
            if key in {"bundle_root", "manifest_path"}:
                continue
            resolved = Path(value).resolve(strict=False)
            try:
                portable_outputs[key] = resolved.relative_to(final_root).as_posix()
            except ValueError as exc:
                raise ValueError(f"Published output is outside the render bundle: {resolved}") from exc

        artifacts: list[dict[str, int | str]] = []
        for artifact in sorted(staging_root.rglob("*")):
            if artifact == path or artifact == staging_root / ".dnadesign-publication-owner.json":
                continue
            if artifact.is_symlink() or not artifact.is_file():
                continue
            artifacts.append(
                {
                    "path": artifact.relative_to(staging_root).as_posix(),
                    "sha256": _sha256_file(artifact),
                    "bytes": artifact.stat().st_size,
                }
            )

        sources = {label: evidence.portable() for label, evidence in self._source_evidence.items()}
        payload = {
            "schema": "dnadesign.baserender.render_bundle_manifest.v1",
            "job_name": self.job_name,
            "sources": sources,
            "counts": {
                "total_rows_seen": self.total_rows_seen,
                "yielded_records": self.yielded_records,
                "skipped_rows_by_reason": self.skipped_rows_by_reason,
                "skipped_records_by_reason": self.skipped_records_by_reason,
                "missing_selection_keys": self.missing_selection_keys,
            },
            "outputs": portable_outputs,
            "output_metrics": self.output_metrics,
            "artifact_inventory": {
                "scope": "all_published_regular_files_except_this_manifest",
                "artifacts": artifacts,
            },
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
