"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/cli_commands/tooling.py

Tooling command handlers for densegen repair, legacy conversion, and demo mocks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..dataset import Dataset
from ..mock import MockSpec


@dataclass(frozen=True)
class ToolingDeps:
    resolve_dataset_name_interactive: Callable[[Path, str | None, bool], str | None]
    resolve_path_anywhere: Callable[[Path], Path]
    create_mock_dataset: Callable[..., int]
    add_demo_columns: Callable[..., int]
    dataset_factory: Callable[[Path, str], Dataset]


def cmd_repair_densegen(args, *, deps: ToolingDeps) -> None:
    from ..convert_legacy import repair_densegen_used_tfbs

    ds_name = deps.resolve_dataset_name_interactive(
        args.root, getattr(args, "dataset", None), bool(getattr(args, "rich", False))
    )
    if not ds_name:
        return
    stats = repair_densegen_used_tfbs(
        dataset_root=args.root,
        dataset_name=ds_name,
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


def cmd_convert_legacy(args, *, deps: ToolingDeps) -> None:
    from ..convert_legacy import convert_legacy, profile_60bp_dual_promoter

    in_paths = [deps.resolve_path_anywhere(p) for p in args.paths]

    stats = convert_legacy(
        dataset_root=args.root,
        dataset_name=args.dataset,
        pt_paths=in_paths,
        profile=profile_60bp_dual_promoter(),
        expected_length=args.expected_length,
        plan_override=args.plan,
        force=bool(args.force),
    )

    msg = f"Converted {stats.rows} row(s) from {stats.files} file(s) into dataset '{args.dataset}'."
    if stats.skipped_bad_len:
        msg += f" Skipped (length≠expected): {stats.skipped_bad_len}."
    print(msg)


def cmd_make_mock(args, *, deps: ToolingDeps) -> None:
    spec = MockSpec(
        n=int(args.n),
        length=int(args.length),
        x_dim=int(args.x_dim),
        y_dim=int(args.y_dim),
        seed=int(args.seed),
        namespace=str(args.namespace),
        csv_path=deps.resolve_path_anywhere(args.from_csv) if args.from_csv else None,
    )
    created = deps.create_mock_dataset(args.root, args.dataset, spec, force=bool(args.force))
    print(
        f"Created mock dataset '{args.dataset}' with {created} rows, "
        f"{spec.namespace}__x_representation[{spec.x_dim}] and {spec.namespace}__label_vec8[{spec.y_dim}]"
        + (" (from CSV)" if args.from_csv else " (random sequences)")
    )
    src = f"--from-csv {spec.csv_path}" if spec.csv_path else f"--length {spec.length}"
    cmd = (
        f"usr make-mock {args.dataset} --n {spec.n} {src} --namespace {spec.namespace} "
        f"--x-dim {spec.x_dim} --y-dim {spec.y_dim}"
    )
    deps.dataset_factory(args.root, args.dataset).append_meta_note("Created mock dataset", cmd)


def cmd_add_demo(args, *, deps: ToolingDeps) -> None:
    n = deps.add_demo_columns(
        args.root,
        args.dataset,
        x_dim=int(args.x_dim),
        y_dim=int(args.y_dim),
        seed=int(args.seed),
        namespace=str(args.namespace),
        allow_overwrite=bool(args.allow_overwrite),
    )
    print(
        f"Added demo columns to {n} rows in '{args.dataset}' "
        f"({args.namespace}__x_representation[{args.x_dim}], {args.namespace}__label_vec8[{args.y_dim}])."
    )
    cmd = f"usr add-demo-cols {args.dataset} --x-dim {args.x_dim} --y-dim {args.y_dim} --namespace {args.namespace}"
    deps.dataset_factory(args.root, args.dataset).append_meta_note("Added demo columns", cmd)
