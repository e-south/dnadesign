"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/cli/commands/tooling/dev.py

Development-only USR tooling commands.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from ....datasets.demo.mock import MockSpec
from .shared import ToolingDeps


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
    source = f"--from-csv {spec.csv_path}" if spec.csv_path else f"--length {spec.length}"
    command = (
        f"USR_SHOW_DEV_COMMANDS=1 usr dev make-mock {args.dataset} --n {spec.n} {source} --namespace {spec.namespace} "
        f"--x-dim {spec.x_dim} --y-dim {spec.y_dim}"
    )
    deps.dataset_factory(args.root, args.dataset).append_meta_note("Created mock dataset", command)


def cmd_add_demo(args, *, deps: ToolingDeps) -> None:
    row_count = deps.add_demo_columns(
        args.root,
        args.dataset,
        x_dim=int(args.x_dim),
        y_dim=int(args.y_dim),
        seed=int(args.seed),
        namespace=str(args.namespace),
        allow_overwrite=bool(args.allow_overwrite),
    )
    print(
        f"Added demo columns to {row_count} rows in '{args.dataset}' "
        f"({args.namespace}__x_representation[{args.x_dim}], {args.namespace}__label_vec8[{args.y_dim}])."
    )
    command = (
        f"USR_SHOW_DEV_COMMANDS=1 usr dev add-demo-cols {args.dataset} "
        f"--x-dim {args.x_dim} --y-dim {args.y_dim} --namespace {args.namespace}"
    )
    deps.dataset_factory(args.root, args.dataset).append_meta_note("Added demo columns", command)
