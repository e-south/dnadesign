"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/cli_commands/tooling/densegen.py

DenseGen-specific USR tooling commands.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .shared import ToolingDeps


def cmd_repair_densegen(args, *, deps: ToolingDeps) -> None:
    from ...convert_legacy import repair_densegen_used_tfbs

    dataset_name = deps.resolve_dataset_name_interactive(
        args.root,
        getattr(args, "dataset", None),
        bool(getattr(args, "rich", False)),
    )
    if not dataset_name:
        return
    stats = repair_densegen_used_tfbs(
        dataset_root=args.root,
        dataset_name=dataset_name,
        min_tfbs_len=int(getattr(args, "min_tfbs_len", 6)),
        dry_run=bool(getattr(args, "dry_run", False)),
        assume_yes=bool(getattr(args, "yes", False)),
        dedupe_policy=(None if getattr(args, "dedupe", "off") == "off" else getattr(args, "dedupe")),
        drop_missing_used_tfbs=bool(getattr(args, "drop_missing_used_tfbs", False)),
        drop_single_tf=bool(getattr(args, "drop_single_tf", False)),
        drop_id_seq_only=bool(getattr(args, "drop_id_seq_only", False)),
        filter_single_tf=bool(getattr(args, "filter_single_tf", False)),
    )
    print(
        f"[repair-densegen] rows={stats.rows_total}  touched={stats.rows_touched}  "
        f"changed(parts/used/detail/counts/u_list)={stats.rows_changed_tfbs_parts}/"
        f"{stats.rows_changed_used_tfbs}/{stats.rows_changed_used_detail}/"
        f"{stats.rows_changed_used_counts}/{stats.rows_changed_used_list}"
    )
