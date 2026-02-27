"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/convert_legacy_dedupe.py

Case-insensitive sequence dedupe helpers for legacy densegen repair flows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Callable

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from .errors import ValidationError


def _sorted_duplicate_groups(table: pa.Table) -> list[tuple[str, pd.DataFrame]]:
    required = {"id", "bio_type", "sequence", "created_at"}
    missing = required - set(table.column_names)
    if missing:
        raise ValidationError(f"Missing required columns for sequence de-duplication: {', '.join(sorted(missing))}")

    frame = table.select(["id", "bio_type", "sequence", "created_at"]).to_pandas()
    frame["_key"] = frame["bio_type"].str.lower() + "|" + frame["sequence"].str.upper()
    groups = frame.groupby("_key").agg({"id": "count"})
    duplicate_keys = groups[groups["id"] > 1].index.tolist()
    if not duplicate_keys:
        return []
    duplicate_frame = frame[frame["_key"].isin(duplicate_keys)]
    return [(key, group.copy()) for key, group in duplicate_frame.groupby("_key")]


def _extend_drop_ids_for_group(
    group: pd.DataFrame,
    *,
    dedupe_policy: str,
    dry_run: bool,
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
) -> list[str]:
    sorted_group = group.sort_values(["created_at", "id"], ascending=True, kind="stable")
    if dedupe_policy == "keep-last":
        sorted_group = group.sort_values(["created_at", "id"], ascending=False, kind="stable")

    if dedupe_policy == "ask" and not dry_run:
        key = str(group.iloc[0]["_key"])
        output_fn(f"\nduplicate sequence (casefold): {key.split('|', 1)[1]}")
        for idx, row in enumerate(sorted_group.reset_index(drop=True).itertuples(index=False), start=1):
            output_fn(f"  {idx}: id={row.id}  created_at={row.created_at}")
        answer = input_fn("Keep which row? [1..n], 0=drop all, s=skip group: ").strip().lower()
        if answer in {"s", "skip"}:
            return []
        if answer in {"0", "drop-all"}:
            return sorted_group["id"].tolist()
        try:
            keep_index = int(answer)
        except ValueError as exc:
            raise ValidationError(f"Invalid selection '{answer}' for de-duplication.") from exc
        if 1 <= keep_index <= len(sorted_group):
            drop_group = sorted_group.drop(sorted_group.index[keep_index - 1])
            return drop_group["id"].tolist()
        raise ValidationError(f"Selection out of range for de-duplication: {keep_index}.")

    return sorted_group.iloc[1:]["id"].tolist()


def apply_casefold_sequence_dedupe(
    table: pa.Table,
    *,
    dedupe_policy: str | None,
    dry_run: bool,
    assume_yes: bool,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> pa.Table:
    if not dedupe_policy or dedupe_policy.lower() == "off":
        return table

    duplicate_groups = _sorted_duplicate_groups(table)
    if not duplicate_groups:
        output_fn("[repair-densegen] dedupe: OK — no case-insensitive duplicates found.")
        return table

    drop_ids: list[str] = []
    for _, group in duplicate_groups:
        drop_ids.extend(
            _extend_drop_ids_for_group(
                group,
                dedupe_policy=dedupe_policy,
                dry_run=dry_run,
                input_fn=input_fn,
                output_fn=output_fn,
            )
        )

    output_fn(f"[repair-densegen] dedupe plan: groups={len(duplicate_groups)}  would_drop={len(drop_ids)}")
    if dry_run or not drop_ids:
        return table
    if not assume_yes:
        answer = input_fn("Proceed with de-duplication? [y/N]: ").strip().lower()
        if answer not in {"y", "yes"}:
            output_fn("Skipping de-duplication.")
            return table

    drop_set = set(drop_ids)
    mask = pc.is_in(table.column("id"), value_set=pa.array(list(drop_set)))
    out = table.filter(pc.invert(mask))
    output_fn(f"[repair-densegen] dedupe: dropped {len(drop_set)} row(s); rows now {out.num_rows}.")
    return out
