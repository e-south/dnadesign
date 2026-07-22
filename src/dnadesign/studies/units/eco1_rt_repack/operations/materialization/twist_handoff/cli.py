"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/twist_handoff/cli.py

CLI for the Eco1 RT Twist full-CDS handoff.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pipeline import materialize_twist_handoff


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize the Eco1 RT Twist full-CDS handoff.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--candidate-selection-panel", type=Path)
    parser.add_argument("--candidate-pool", type=Path)
    parser.add_argument("--foldcheck-fasta", type=Path)
    parser.add_argument("--generation-policy-positions", type=Path)
    parser.add_argument("--wild-type-genbank", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = materialize_twist_handoff(
            repo_root=args.repo_root,
            output_root=args.output_root,
            candidate_selection_panel_path=args.candidate_selection_panel,
            candidate_pool_path=args.candidate_pool,
            foldcheck_fasta_path=args.foldcheck_fasta,
            generation_policy_positions_path=args.generation_policy_positions,
            wild_type_genbank_path=args.wild_type_genbank,
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "manifest_path": str(result.manifest_path),
                "twist_csv_path": str(result.twist_csv_path),
                "fasta_path": str(result.fasta_path),
                "genbank_paths": [str(path) for path in result.genbank_paths],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0
