"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/cli_commands/tooling/legacy.py

Legacy-import USR tooling commands.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .shared import ToolingDeps


def cmd_convert_legacy(args, *, deps: ToolingDeps) -> None:
    from ...convert_legacy import convert_legacy, profile_60bp_dual_promoter

    input_paths = [deps.resolve_path_anywhere(path) for path in args.paths]

    stats = convert_legacy(
        dataset_root=args.root,
        dataset_name=args.dataset,
        pt_paths=input_paths,
        profile=profile_60bp_dual_promoter(),
        expected_length=args.expected_length,
        plan_override=args.plan,
        force=bool(args.force),
    )

    message = f"Converted {stats.rows} row(s) from {stats.files} file(s) into dataset '{args.dataset}'."
    if stats.skipped_bad_len:
        message += f" Skipped (length≠expected): {stats.skipped_bad_len}."
    print(message)
