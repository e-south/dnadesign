"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/dataset_events.py

Dataset-scoped wrappers for USR event logging.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .events import record_event


def record_dataset_event(
    *,
    events_path: Path,
    action: str,
    dataset_name: str,
    dataset_root: Path,
    target_path: Path,
    args: Optional[dict] = None,
    metrics: Optional[dict] = None,
    artifacts: Optional[dict] = None,
    maintenance: Optional[dict] = None,
    registry_hash: Optional[str] = None,
    actor: Optional[dict] = None,
) -> None:
    record_event(
        events_path,
        action,
        dataset=dataset_name,
        args=args,
        metrics=metrics,
        artifacts=artifacts,
        maintenance=maintenance,
        target_path=target_path,
        dataset_root=dataset_root,
        registry_hash=registry_hash,
        actor=actor,
    )
