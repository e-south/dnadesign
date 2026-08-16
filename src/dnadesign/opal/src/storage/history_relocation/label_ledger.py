"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/storage/history_relocation/label_ledger.py

Stages append-only label events when campaign histories are consolidated.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from ...core.utils import OpalError, file_sha256
from ..parquet_io import read_parquet_df, write_parquet_df
from .contracts import CampaignHistory, HistoryRelocationPlan
from .inspection import canonical_sha256, jsonable

_EVENT_KEY_COLUMNS = ("id", "observed_round")


def label_ledger_parts(history: CampaignHistory) -> tuple[Path, ...]:
    root = history.workdir / "outputs" / "ledger" / "labels.parquet"
    return tuple(sorted(root.rglob("*.parquet"))) if root.is_dir() else ()


def _event_key(row: dict[str, object]) -> tuple[str, int]:
    return str(row["id"]), int(row["observed_round"])


def _label_frames(
    history: CampaignHistory,
    *,
    label: str,
) -> tuple[list[tuple[Path, pd.DataFrame]], dict[tuple[str, int], str]]:
    frames: list[tuple[Path, pd.DataFrame]] = []
    event_digests: dict[tuple[str, int], str] = {}
    for part in label_ledger_parts(history):
        frame = read_parquet_df(part)
        missing = sorted(set(_EVENT_KEY_COLUMNS) - set(frame.columns))
        if missing:
            raise OpalError(f"{label} label ledger part is missing event-key columns {missing}: {part}.")
        frames.append((part, frame))
        for record in frame.to_dict(orient="records"):
            key = _event_key(record)
            if key in event_digests:
                raise OpalError(f"{label} label ledger contains duplicate immutable event key {key!r}.")
            event_digests[key] = canonical_sha256(jsonable(record))
    return frames, event_digests


def stage_label_ledger(
    plan: HistoryRelocationPlan,
    *,
    staging_root: Path,
) -> list[tuple[Path, Path]]:
    source_root = plan.source.workdir / "outputs" / "ledger" / "labels.parquet"
    target_root = plan.target.workdir / "outputs" / "ledger" / "labels.parquet"
    source_frames, source_events = _label_frames(plan.source, label="Source campaign")
    _, target_events = _label_frames(plan.target, label="Target campaign")
    for key in sorted(set(source_events) & set(target_events)):
        if source_events[key] != target_events[key]:
            raise OpalError(f"Campaign histories contain a conflicting immutable label event for key {key!r}.")
    moves: list[tuple[Path, Path]] = []
    destinations: set[Path] = set()
    for source_part, source_frame in source_frames:
        new_rows = [
            _event_key(record) not in target_events
            for record in source_frame.loc[:, list(_EVENT_KEY_COLUMNS)].to_dict(orient="records")
        ]
        if not any(new_rows):
            continue
        staged_frame = source_frame.loc[new_rows].reset_index(drop=True)
        relative = source_part.relative_to(source_root)
        target_part = target_root / relative
        source_digest = file_sha256(source_part)
        if target_part.is_file():
            if file_sha256(target_part) == source_digest:
                continue
            target_part = target_root / f"part-history-{source_part.stem}-{source_digest[:16]}.parquet"
        if target_part in destinations:
            raise OpalError(f"Label ledger relocation produces duplicate destination {target_part}.")
        if target_part.exists():
            if target_part.is_file() and file_sha256(target_part) == source_digest:
                continue
            raise OpalError(f"Label ledger relocation destination already exists: {target_part}.")
        staged_part = staging_root / target_part.relative_to(plan.target.workdir)
        staged_part.parent.mkdir(parents=True, exist_ok=True)
        if len(staged_frame) == len(source_frame):
            shutil.copy2(source_part, staged_part)
        else:
            write_parquet_df(staged_part, staged_frame, index=False)
        destinations.add(target_part)
        moves.append((staged_part, target_part))
    return moves


__all__ = ["label_ledger_parts", "stage_label_ledger"]
