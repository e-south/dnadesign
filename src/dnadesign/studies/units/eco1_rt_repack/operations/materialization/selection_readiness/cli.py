"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/selection_readiness/cli.py

CLI for Eco1 panel-selection materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.constants import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SOURCE_OUTPUT_ROOT,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.pipeline import (
    materialize_selection_readiness,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the panel-selection materialization CLI parser."""

    parser = argparse.ArgumentParser(description="Materialize Eco1 RT panel-selection artifacts.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--source-output-root", type=Path, default=DEFAULT_SOURCE_OUTPUT_ROOT)
    parser.add_argument("--selection-root", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run panel-selection materialization and print emitted paths."""

    args = build_parser().parse_args(argv)
    try:
        result = materialize_selection_readiness(
            repo_root=args.repo_root,
            output_root=args.output_root,
            source_output_root=args.source_output_root,
            selection_root=args.selection_root,
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "feasibility_report_path": str(result.feasibility_report_path),
                "candidate_triage_table_path": str(result.candidate_triage_table_path),
                "candidate_selection_panel_path": str(result.candidate_selection_panel_path),
                "candidate_handoff_sequence_csv_path": str(result.candidate_handoff_sequence_csv_path),
                "selection_readiness_manifest_path": str(result.manifest_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0
