"""
Sequence-view maintenance helpers.

These helpers repair sidecar-level metadata without mutating canonical USR
records or feature sidecars. They are intentionally explicit maintenance tools,
not fallbacks in the sequence-view writer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..dataset import Dataset
from ..storage.locking import dataset_write_lock
from .models import SequenceViewRecord
from .store import _rows_to_table, _write_sequence_views_atomic, load_sequence_views, sequence_views_path


@dataclass(frozen=True)
class SequenceViewAliasConflictExample:
    alias: str
    view_ids: list[str]


@dataclass(frozen=True)
class SequenceViewAliasRepairResult:
    dataset: str
    path: str
    rows_total: int
    duplicate_alias_keys: int
    conflicting_view_rows: int
    aliases_removed: int
    rows_touched: int
    written: bool
    examples: list[SequenceViewAliasConflictExample] = field(default_factory=list)


def _alias_key(value: str) -> str:
    return value.casefold()


def _alias_conflicts(rows: list[SequenceViewRecord]) -> dict[str, set[str]]:
    alias_to_view_ids: dict[str, set[str]] = {}
    for row in rows:
        for alias in row.aliases or []:
            alias_to_view_ids.setdefault(_alias_key(alias), set()).add(str(row.view_id))
    return {alias: view_ids for alias, view_ids in alias_to_view_ids.items() if len(view_ids) > 1}


def _conflict_examples(
    rows: list[SequenceViewRecord],
    conflicts: dict[str, set[str]],
    *,
    limit: int,
) -> list[SequenceViewAliasConflictExample]:
    display_by_key: dict[str, str] = {}
    for row in rows:
        for alias in row.aliases or []:
            display_by_key.setdefault(_alias_key(alias), alias)
    examples: list[SequenceViewAliasConflictExample] = []
    for alias_key in sorted(conflicts)[:limit]:
        examples.append(
            SequenceViewAliasConflictExample(
                alias=display_by_key.get(alias_key, alias_key),
                view_ids=sorted(conflicts[alias_key]),
            )
        )
    return examples


def repair_sequence_view_alias_conflicts(
    dataset: Dataset,
    *,
    write: bool = False,
    example_limit: int = 20,
    actor: dict[str, object] | None = None,
) -> SequenceViewAliasRepairResult:
    """Remove non-unique aliases from a sequence-view sidecar.

    The repair policy is conservative by design: any alias that resolves to more
    than one ``view_id`` is removed from every conflicting row. This avoids
    choosing a winner for ambiguous human aliases while preserving all stable
    view IDs, view names, lineage, bounds, and feature sidecars.
    """

    dataset._require_exists()  # noqa: SLF001
    path = sequence_views_path(dataset)
    rows = load_sequence_views(dataset)
    view_ids = [str(row.view_id) for row in rows]
    if len(view_ids) != len(set(view_ids)):
        raise ValueError("Cannot repair sequence-view aliases while duplicate view_id rows exist.")

    conflicts = _alias_conflicts(rows)
    conflicting_aliases = set(conflicts)
    touched = 0
    removed = 0
    repaired: list[SequenceViewRecord] = []
    conflicting_view_ids = {view_id for view_ids_for_alias in conflicts.values() for view_id in view_ids_for_alias}
    for row in rows:
        aliases = list(row.aliases or [])
        if not aliases:
            repaired.append(row)
            continue
        kept = [alias for alias in aliases if _alias_key(alias) not in conflicting_aliases]
        removed_here = len(aliases) - len(kept)
        if removed_here:
            touched += 1
            removed += removed_here
            repaired.append(row.model_copy(update={"aliases": kept or None}))
        else:
            repaired.append(row)

    if write and removed:
        with dataset_write_lock(dataset.dir):
            _write_sequence_views_atomic(path, _rows_to_table(repaired))
            dataset._record_event(  # noqa: SLF001
                "repair_sequence_view_alias_conflicts",
                args={
                    "policy": "drop_all_non_unique_aliases",
                    "duplicate_alias_keys": len(conflicts),
                    "aliases_removed": removed,
                    "rows_touched": touched,
                },
                metrics={
                    "duplicate_alias_keys": len(conflicts),
                    "conflicting_view_rows": len(conflicting_view_ids),
                    "aliases_removed": removed,
                    "rows_touched": touched,
                },
                target_path=path,
                actor=actor,
            )

    return SequenceViewAliasRepairResult(
        dataset=dataset.name,
        path=str(path),
        rows_total=len(rows),
        duplicate_alias_keys=len(conflicts),
        conflicting_view_rows=len(conflicting_view_ids),
        aliases_removed=removed,
        rows_touched=touched,
        written=bool(write and removed),
        examples=_conflict_examples(rows, conflicts, limit=example_limit),
    )
