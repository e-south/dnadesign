"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/measured_reader_vec8/cli.py

Provides the measured Reader vec8 staging command.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .staging import build_measured_reader_vec8_staging, write_measured_reader_vec8_batch0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build stress-study measured reader vec8 batch0 staging inputs.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--reader-root", type=Path, default=Path.cwd().parent / "reader")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if args.write:
        result = write_measured_reader_vec8_batch0(
            repo_root=args.repo_root,
            reader_root=args.reader_root,
            out_dir=args.out_dir,
            overwrite=args.overwrite,
        )
        payload = result.to_dict()
    else:
        staging = build_measured_reader_vec8_staging(repo_root=args.repo_root, reader_root=args.reader_root)
        rows_by_campaign = staging.measured_frame.groupby("campaign_slug").size().to_dict()
        payload = {
            "schema_version": "stress_ethanol_cipro_growth.measured_reader_vec8.preview.v1",
            "summary": staging.summary,
            "measured_rows_by_campaign": {str(key): int(value) for key, value in rows_by_campaign.items()},
        }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


__all__ = ["main"]
