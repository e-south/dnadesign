"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/contracts/workspace_storage/cli.py

Command-line verification for explicit workspace-storage roots.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .models import WorkspaceStorageError
from .validation import verify_workspace_storage


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dnadesign-workspace-storage")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="verify one explicit workspace root")
    validate.add_argument("workspace_root", type=Path)
    validate.add_argument("--json", action="store_true", dest="json_output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        verified = verify_workspace_storage(args.workspace_root)
    except WorkspaceStorageError as exc:
        print(f"workspace storage validation failed: {exc}", file=sys.stderr)
        return 2
    summary = verified.summary()
    if args.json_output:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(
            f"verified {summary['workspace_id']} "
            f"({summary['input_count']} inputs, {summary['artifact_count']} artifacts)"
        )
    return 0
