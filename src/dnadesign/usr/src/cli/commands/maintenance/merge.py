"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/cli/commands/maintenance/merge.py

USR CLI maintenance merge command implementations.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ....contracts import SequencesError
from ....dataset import Dataset


@dataclass(frozen=True)
class MergeDeps:
    resolve_merge_policy: Callable[[str], object]
    merge_usr_to_usr: Callable[..., object]
    mode_require_same: object
    mode_union: object
    dataset_factory: Callable[[Path, str], Dataset]


def cmd_merge_datasets(args, *, deps: MergeDeps) -> None:
    columns = str(getattr(args, "columns", "") or "")
    cols_subset = [c.strip() for c in columns.split(",") if c.strip()] if columns else None
    require_same = bool(getattr(args, "require_same", False))
    union_columns = bool(getattr(args, "union_columns", False))
    if require_same == union_columns:
        raise SequencesError("Choose exactly one of --require-same-columns or --union-columns.")
    mode = deps.mode_require_same if require_same else deps.mode_union
    dup_policy = str(getattr(args, "dup_policy", "error") or "error")
    policy = deps.resolve_merge_policy(dup_policy)
    coerce_overlap = str(getattr(args, "coerce_overlap", "none") or "none")
    if coerce_overlap not in {"none", "to-dest"}:
        raise SequencesError("--coerce-overlap must be one of: none, to-dest")
    dry_run = bool(getattr(args, "dry_run", False))
    assume_yes = bool(getattr(args, "yes", False))
    note = str(getattr(args, "note", "") or "")
    carry_namespaces = [
        str(value).strip() for value in list(getattr(args, "carry_namespaces", []) or []) if str(value).strip()
    ]
    if hasattr(args, "avoid_casefold_dups"):
        avoid_casefold_dups = bool(getattr(args, "avoid_casefold_dups", True))
    else:
        avoid_casefold_dups = not bool(getattr(args, "no_avoid_casefold_dups", False))

    ds_dest = deps.dataset_factory(args.root, args.dest)
    with ds_dest.maintenance(reason="merge"):
        preview = deps.merge_usr_to_usr(
            root=args.root,
            dest=args.dest,
            src=args.src,
            columns_mode=mode,
            duplicate_policy=policy,
            columns_subset=cols_subset,
            dry_run=dry_run,
            assume_yes=assume_yes,
            note=note,
            overlap_coercion=("to-dest" if coerce_overlap == "to-dest" else "none"),
            avoid_casefold_dups=avoid_casefold_dups,
            carry_namespaces=carry_namespaces,
        )

    action = "DRY-RUN" if dry_run else "MERGED"
    message = (
        f"[{action}] dest='{args.dest}'  src='{args.src}'  "
        f"rows_src={preview.src_rows}  "
        f"duplicates_total={preview.duplicates_total}  "
        f"duplicates_skipped={preview.duplicates_skipped}  "
        f"duplicates_replaced={preview.duplicates_replaced}  "
        f"rows_added={preview.new_rows}  "
        f"dest_rows: {preview.dest_rows_before} -> {preview.dest_rows_after}  "
        f"columns(total={preview.columns_total}, overlap={preview.overlapping_columns})  "
        f"dup_policy={preview.duplicate_policy.value}"
    )
    if preview.carried_namespace_counts:
        carried = ",".join(f"{namespace}:{rows}" for namespace, rows in preview.carried_namespace_counts)
        message += f"  carried({carried})"
    print(message)
    if not dry_run:
        cmd = (
            f"usr merge-datasets --dest {args.dest} --src {args.src} "
            f"{'--require-same-columns' if require_same else '--union-columns'} "
            f"--if-duplicate {dup_policy}"
        )
        if carry_namespaces:
            cmd += "".join(f" --carry-namespace {namespace}" for namespace in carry_namespaces)
        ds_dest.append_meta_note(
            f"Merged from '{args.src}' -> '{args.dest}' (added {preview.new_rows} rows; dup_policy={preview.duplicate_policy.value})",  # noqa: E501
            cmd,
        )
