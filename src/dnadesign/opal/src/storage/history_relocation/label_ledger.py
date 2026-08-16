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

from ...core.utils import OpalError, file_sha256
from .contracts import CampaignHistory, HistoryRelocationPlan


def label_ledger_parts(history: CampaignHistory) -> tuple[Path, ...]:
    root = history.workdir / "outputs" / "ledger" / "labels.parquet"
    return tuple(sorted(root.rglob("*.parquet"))) if root.is_dir() else ()


def stage_label_ledger(
    plan: HistoryRelocationPlan,
    *,
    staging_root: Path,
) -> list[tuple[Path, Path]]:
    source_root = plan.source.workdir / "outputs" / "ledger" / "labels.parquet"
    target_root = plan.target.workdir / "outputs" / "ledger" / "labels.parquet"
    moves: list[tuple[Path, Path]] = []
    destinations: set[Path] = set()
    for source_part in label_ledger_parts(plan.source):
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
        shutil.copy2(source_part, staged_part)
        destinations.add(target_part)
        moves.append((staged_part, target_part))
    return moves


__all__ = ["label_ledger_parts", "stage_label_ledger"]
