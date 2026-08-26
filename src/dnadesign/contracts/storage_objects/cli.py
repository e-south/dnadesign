"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/contracts/storage_objects/cli.py

Command-line inventory and verification for external storage objects.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .inventory import inventory_storage_object, refresh_storage_object
from .models import ObjectKind, RetentionPolicy, StorageClass, StorageObjectError
from .validation import verify_storage_object, verify_storage_root


def _add_json(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", dest="json_output")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dnadesign-storage")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser("inventory", help="create and verify one exact manifest")
    inventory.add_argument("storage_root", type=Path)
    inventory.add_argument("--storage-id", required=True)
    inventory.add_argument("--owner-repository", required=True)
    inventory.add_argument("--owner-tool", required=True)
    inventory.add_argument("--object-kind", choices=[value.value for value in ObjectKind], required=True)
    inventory.add_argument("--content-schema", required=True)
    inventory.add_argument("--content-schema-version", required=True)
    inventory.add_argument("--producer-revision", required=True)
    inventory.add_argument("--storage-class", choices=[value.value for value in StorageClass], required=True)
    inventory.add_argument("--retention-policy", choices=[value.value for value in RetentionPolicy], required=True)
    inventory.add_argument("--input", action="append", default=[])
    inventory.add_argument("--metadata", action="append", default=[])
    inventory.add_argument("--original-execution-path")
    inventory.add_argument("--demo", action="store_true")
    _add_json(inventory)

    refresh = subparsers.add_parser("refresh", help="refresh one changed object using its prior receipt digest")
    refresh.add_argument("storage_root", type=Path)
    refresh.add_argument("--expected-manifest-digest", required=True)
    _add_json(refresh)

    validate = subparsers.add_parser("validate", help="verify one explicit storage object")
    validate.add_argument("storage_root", type=Path)
    _add_json(validate)

    validate_root = subparsers.add_parser("validate-root", help="verify every convention-routed object under storage")
    validate_root.add_argument("storage_root", type=Path)
    _add_json(validate_root)
    return parser


def _print(summary: dict[str, object], *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(json.dumps(summary, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "inventory":
            summary = inventory_storage_object(
                args.storage_root,
                storage_id=args.storage_id,
                owner_repository=args.owner_repository,
                owner_tool=args.owner_tool,
                object_kind=args.object_kind,
                content_schema=args.content_schema,
                content_schema_version=args.content_schema_version,
                producer_revision=args.producer_revision,
                storage_class=args.storage_class,
                retention_policy=args.retention_policy,
                input_paths=tuple(args.input),
                metadata_paths=tuple(args.metadata),
                original_execution_path=args.original_execution_path,
                demo=args.demo,
            )
        elif args.command == "refresh":
            summary = refresh_storage_object(
                args.storage_root,
                expected_manifest_digest=args.expected_manifest_digest,
            )
        elif args.command == "validate":
            summary = verify_storage_object(args.storage_root).summary()
        else:
            summary = verify_storage_root(args.storage_root).summary()
    except StorageObjectError as exc:
        print(f"storage validation failed: {exc}", file=sys.stderr)
        return 2
    _print(summary, json_output=args.json_output)
    return 0
