"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/sae_window_summary/cli.py

CLI for Eco1 SAE window-summary materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.sae_window_summary.constants import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_REPORT_ROOT,
    DEFAULT_SOURCE_OUTPUT_ROOT,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.sae_window_summary.pipeline import (
    materialize_sae_window_summary,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the SAE window-summary materialization CLI parser."""

    parser = argparse.ArgumentParser(description="Materialize Eco1 Biohub ESMC SAE window summaries.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--source-output-root", type=Path, default=DEFAULT_SOURCE_OUTPUT_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--residue-features-path", type=Path, default=None)
    parser.add_argument("--profile-path", type=Path, default=None)
    parser.add_argument("--feature-catalog-path", type=Path, default=None)
    parser.add_argument("--candidate-pool-path", type=Path, default=None)
    parser.add_argument("--mask-set-path", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run SAE window-summary materialization and print emitted paths."""

    args = build_parser().parse_args(argv)
    try:
        result = materialize_sae_window_summary(
            repo_root=args.repo_root,
            output_root=args.output_root,
            source_output_root=args.source_output_root,
            report_root=args.report_root,
            residue_features_path=args.residue_features_path,
            profile_path=args.profile_path,
            feature_catalog_path=args.feature_catalog_path,
            candidate_pool_path=args.candidate_pool_path,
            mask_set_path=args.mask_set_path,
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "sae_window_summary_path": str(result.summary_path),
                "sae_window_manifest_path": str(result.manifest_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0
