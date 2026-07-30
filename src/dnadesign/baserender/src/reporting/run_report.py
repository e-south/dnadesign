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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


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

    def write_portable_manifest(self, path: Path, *, bundle_root: Path, staging_root: Path) -> None:
        """Write the portable, complete catalog for one immutable render bundle."""
        final_root = bundle_root.resolve(strict=False)

        def _source_artifact(source: str | None) -> dict[str, int | str] | None:
            if source is None:
                return None
            source_path = Path(source)
            if not source_path.is_file() or source_path.is_symlink():
                raise ValueError(f"Render source is unavailable or unsafe: {source_path}")
            return {
                "sha256": _sha256_file(source_path),
                "bytes": source_path.stat().st_size,
            }

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
            if artifact == path or artifact.name == ".dnadesign-publication-owner.json":
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

        sources = {"input": _source_artifact(self.input_path)}
        selection = _source_artifact(self.selection_path)
        if selection is not None:
            sources["selection"] = selection
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
                "scope": "all_regular_files_except_this_manifest",
                "artifacts": artifacts,
            },
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
