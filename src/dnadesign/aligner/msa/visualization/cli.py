"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/aligner/msa/visualization/cli.py

Command-line interface for generic MSA visualization sidecars.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from dnadesign.aligner.msa.visualization import MsaVisualizationRequest
from dnadesign.aligner.msa.visualization.materialization import (
    materialize_msa_visualizations,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the visualization CLI parser."""

    parser = argparse.ArgumentParser(
        description="Materialize generic MSA QC and visualization sidecars.",
    )
    parser.add_argument("--alignment-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--profile-id", action="append", required=True)
    parser.add_argument("--target-row-id", required=True)
    parser.add_argument("--target-sequence-hash")
    parser.add_argument("--annotation-tracks-yaml", type=Path)
    parser.add_argument("--exemplar-rows-yaml", type=Path)
    parser.add_argument("--panel-spec-yaml", type=Path)
    parser.add_argument("--allow-missing-profiles", action="store_true")
    parser.add_argument("--created-at", default="unknown")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the visualization materializer."""

    args = build_parser().parse_args(argv)
    result = materialize_msa_visualizations(
        MsaVisualizationRequest(
            alignment_root=args.alignment_root,
            output_root=args.output_root,
            profile_ids=tuple(args.profile_id),
            target_row_id=args.target_row_id,
            target_sequence_hash=args.target_sequence_hash,
            annotation_tracks_yaml=args.annotation_tracks_yaml,
            exemplar_rows_yaml=args.exemplar_rows_yaml,
            panel_spec_yaml=args.panel_spec_yaml,
            allow_missing_profiles=args.allow_missing_profiles,
            created_at=args.created_at,
        )
    )
    print(
        json.dumps(
            {
                "index_manifest_path": str(result.index_manifest_path),
                "position_qc_csv_path": str(result.position_qc_csv_path),
                "html_report_path": str(result.html_report_path),
                "profile_ids": list(result.profile_ids),
                "missing_profile_ids": list(result.missing_profile_ids),
                "profile_exemplar_svg_paths": {
                    profile_id: str(path) for profile_id, path in result.profile_exemplar_svg_paths.items()
                },
                "profile_alignment_overview_svg_paths": {
                    profile_id: str(path) for profile_id, path in result.profile_alignment_overview_svg_paths.items()
                },
                "profile_consensus_histogram_svg_paths": {
                    profile_id: str(path) for profile_id, path in result.profile_consensus_histogram_svg_paths.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0
