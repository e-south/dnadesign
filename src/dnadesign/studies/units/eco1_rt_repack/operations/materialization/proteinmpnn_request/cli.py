"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/proteinmpnn_request/cli.py

CLI for Eco1 ProteinMPNN request materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.proteinmpnn_request.pipeline import (
    materialize_proteinmpnn_request,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the request-adapter CLI parser."""

    parser = argparse.ArgumentParser(
        description="Materialize ProteinMPNN helper-compatible request sidecars from Eco1 thread_plan.yaml."
    )
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run request materialization and print emitted paths as JSON."""

    args = build_parser().parse_args(argv)
    result = materialize_proteinmpnn_request(repo_root=args.repo_root, output_root=args.output_root)
    print(
        json.dumps(
            {
                "chain_a_backbone_pdb_path": str(result.chain_a_backbone_pdb_path),
                "parsed_pdbs_path": str(result.parsed_pdbs_path),
                "assigned_chains_path": str(result.assigned_chains_path),
                "fixed_positions_path": str(result.fixed_positions_path),
                "request_manifest_path": str(result.request_manifest_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
