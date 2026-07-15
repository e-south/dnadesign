"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_window_label_promotion/cli.py

Publish and verify immutable OPAL labels from approved study observations.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations.aggregation import (
    VALUE_COLUMNS,
)

from .contracts import DEFAULT_CAMPAIGN_CONFIG_PATH, DEFAULT_OUTPUT_DIRECTORY, confined_relative_directory
from .publication import verify_label_bundle
from .publisher import publish_response_window_labels


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish or verify the stress study's immutable response-window OPAL label bundle."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    publish = subparsers.add_parser("publish", help="Create one new versioned label bundle.")
    publish.add_argument("--observation-bundle", type=Path, required=True)
    _add_output_binding(publish)

    verify = subparsers.add_parser("verify", help="Verify an existing versioned label bundle.")
    _add_output_binding(verify)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "publish":
        result = publish_response_window_labels(
            observation_bundle_dir=args.observation_bundle,
            dataset_root=args.dataset_root,
            output_relative_directory=args.output_relative_directory,
            campaign_config_path=args.campaign_config,
        )
        payload: dict[str, object] = {
            "output_directory": str(result.output_directory),
            "label_path": str(result.label_path),
            "study_provenance_path": str(result.study_provenance_path),
            "promotion_manifest_path": str(result.promotion_manifest_path),
            "candidate_count": result.candidate_count,
            "create_only": True,
        }
    else:
        relative = confined_relative_directory(args.output_relative_directory)
        snapshot = verify_label_bundle(
            args.dataset_root.expanduser().resolve(),
            relative_dir=relative,
            expected_width=len(VALUE_COLUMNS),
            campaign_config_path=args.campaign_config,
        )
        payload = {
            "promotion_manifest_path": str(snapshot.promotion.manifest_path),
            "promotion_manifest_sha256": snapshot.promotion.manifest_sha256,
            "label_path": str(snapshot.promotion.label_path),
            "label_sha256": snapshot.promotion.label_sha256,
            "candidate_count": len(snapshot.labels),
            "verified": True,
        }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _add_output_binding(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-relative-directory", default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--campaign-config", type=Path, default=DEFAULT_CAMPAIGN_CONFIG_PATH)


__all__ = ["build_parser", "main"]
