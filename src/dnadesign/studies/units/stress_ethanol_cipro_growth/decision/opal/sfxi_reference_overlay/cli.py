"""Command-line preview and publication for the study-owned SFXI overlay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dnadesign.usr import default_usr_root

from .reader_records import default_selection_path
from .recipe import DEFAULT_OUTPUT_DATASET, build_overlay_preview, publish_overlay


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview the stress-study SFXI reference overlay.")
    parser.add_argument("--usr-root", type=Path, default=default_usr_root())
    parser.add_argument("--dataset", default=DEFAULT_OUTPUT_DATASET)
    parser.add_argument("--reader-root", type=Path, required=True)
    parser.add_argument("--record-selection", type=Path, default=default_selection_path())
    parser.add_argument("--expected-count", type=int, default=23)
    parser.add_argument("--write", action="store_true", help="Publish through USR; default is read-only.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    preview = build_overlay_preview(
        usr_root=args.usr_root,
        dataset_name=args.dataset,
        reader_root=args.reader_root,
        selection_path=args.record_selection,
    )
    if preview.table.num_rows != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} overlay rows, found {preview.table.num_rows}.")
    written = publish_overlay(usr_root=args.usr_root, preview=preview) if args.write else 0
    print(
        json.dumps(
            {
                "dataset": preview.dataset_name,
                "rows": preview.table.num_rows,
                "source_ref": preview.source_ref,
                "record_digests": list(preview.record_digests),
                "written": bool(written),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0
