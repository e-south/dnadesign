"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/dataset_activity.py

Helpers for dataset meta-note IO and dataset-scoped event emission.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .dataset_events import record_dataset_event
from .storage.parquet import now_utc


def append_meta_note(
    *,
    dataset_dir: Path,
    dataset_name: str,
    meta_path: Path,
    title: str,
    code_block: str | None = None,
    timestamp_utc: str | None = None,
) -> None:
    ts = str(timestamp_utc or now_utc())
    dataset_dir.mkdir(parents=True, exist_ok=True)
    if not meta_path.exists():
        header = (
            f"name: {dataset_name}\n"
            f"created_at: {ts}\n"
            "source: \n"
            "notes: \n"
            "schema: USR v1\n\n"
            f"### Updates ({ts.split('T')[0]})\n"
        )
        meta_path.write_text(header, encoding="utf-8")
    with meta_path.open("a", encoding="utf-8") as handle:
        handle.write(f"- {ts}: {title}\n")
        if code_block:
            handle.write("```bash\n")
            handle.write(code_block.strip() + "\n")
            handle.write("```\n")


def record_dataset_activity_event(
    *,
    events_path: Path,
    action: str,
    dataset_name: str,
    dataset_root: Path,
    records_path: Path,
    args: Optional[dict] = None,
    metrics: Optional[dict] = None,
    artifacts: Optional[dict] = None,
    maintenance: Optional[dict] = None,
    target_path: Optional[Path] = None,
    registry_hash: Optional[str] = None,
    actor: Optional[dict] = None,
) -> None:
    record_dataset_event(
        events_path=events_path,
        action=action,
        dataset_name=dataset_name,
        dataset_root=dataset_root,
        target_path=target_path or records_path,
        args=args,
        metrics=metrics,
        artifacts=artifacts,
        maintenance=maintenance,
        registry_hash=registry_hash,
        actor=actor,
    )
