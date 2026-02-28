"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/cli_commands/namespace_handlers.py

Namespace registry command handlers for USR CLI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class NamespaceDeps:
    load_registry: Callable[..., dict]
    parse_columns_spec: Callable[..., list]
    register_namespace: Callable[..., Path]


def cmd_namespace_list(args, *, deps: NamespaceDeps) -> None:
    entries = deps.load_registry(args.root, required=True)
    if not entries:
        print("(no namespaces registered)")
        return
    for name, entry in sorted(entries.items()):
        cols = ", ".join(c.name for c in entry.columns)
        print(f"{name}: {cols}")


def cmd_namespace_show(args, *, deps: NamespaceDeps) -> None:
    entries = deps.load_registry(args.root, required=True)
    if args.name not in entries:
        raise SystemExit(f"Namespace '{args.name}' not registered.")
    entry = entries[args.name]
    print(f"name: {entry.namespace}")
    print(f"owner: {entry.owner or ''}")
    print(f"description: {entry.description or ''}")
    print("columns:")
    for col in entry.columns:
        print(f"  - {col.name}: {col.type}")


def cmd_namespace_register(args, *, deps: NamespaceDeps) -> None:
    cols = deps.parse_columns_spec(args.columns, namespace=args.namespace)
    path = deps.register_namespace(
        args.root,
        namespace=args.namespace,
        columns=cols,
        owner=args.owner,
        description=args.description,
        overwrite=bool(args.overwrite),
    )
    print(f"Registered namespace '{args.namespace}' in {path}.")
