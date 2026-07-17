"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/multistate_behavior_cli.py

Operate the study-owned Multistate Response Behavior shadow bundle.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runtime.multistate_behavior_publication import (
    publish_multistate_behavior_shadow,
    verify_multistate_behavior_shadow,
)
from .runtime.multistate_behavior_shadow import (
    VerifiedMultistateBehaviorShadow,
    load_verified_multistate_behavior_shadow,
)

DEFAULT_OUT_DIR = Path(
    "src/dnadesign/studies/units/stress_ethanol_cipro_growth/"
    "workbench/outputs/multistate_response_behavior_shadow/latest"
)


def main(argv: list[str] | None = None) -> int:
    """Preview, publish, or verify the digest-bound shadow evidence."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("preview", "publish"):
        subparser = subparsers.add_parser(command)
        _add_source_arguments(subparser)
        if command == "publish":
            subparser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
            subparser.add_argument("--overwrite", action="store_true")
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "verify":
        manifest = verify_multistate_behavior_shadow(args.bundle.resolve())
        print(json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True))
        return 0
    preview = load_verified_multistate_behavior_shadow(
        repo_root=args.repo_root.resolve(),
        reader_bundle_root=args.reader_bundle.resolve(),
        candidate_bindings_root=args.candidate_bindings.resolve(),
        prediction_run_id=args.prediction_run_id,
    )
    if args.command == "preview":
        print(json.dumps(_preview_summary(preview), allow_nan=False, indent=2, sort_keys=True))
        return 0
    out_dir = args.out_dir
    if not out_dir.is_absolute():
        out_dir = args.repo_root.resolve() / out_dir
    manifest = publish_multistate_behavior_shadow(
        preview,
        out_dir=out_dir,
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True))
    return 0


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--reader-bundle", type=Path, required=True)
    parser.add_argument("--candidate-bindings", type=Path, required=True)
    parser.add_argument("--prediction-run-id", required=True)


def _preview_summary(preview: VerifiedMultistateBehaviorShadow) -> dict[str, object]:
    receipt = preview.normalization.verified_cohort_receipt
    if receipt is None:
        raise ValueError("behavior shadow preview lacks a verified cohort receipt.")
    return {
        "schema_id": "stress_ethanol_cipro_growth.multistate_response_behavior_shadow_preview.v1",
        "status": "shadow_only",
        "activation": {"campaign": "prohibited", "synthesis": "prohibited"},
        "protocol_id": preview.normalization.protocol.protocol_id,
        "normalization": preview.normalization.normalization,
        "cohort": {
            "unit_count": receipt.unit_count,
            "candidate_count": receipt.candidate_count,
            "reader_experiment_count": receipt.reader_experiment_count,
            "excluded_nonexact_unit_count": receipt.excluded_nonexact_unit_count,
        },
        "prediction": preview.source["prediction"],
        "hard_behavior_comparison": preview.hard_comparison.summary.to_dict(orient="records"),
        "claim_boundary": "shadow_evidence_only_no_campaign_activation_or_synthesis_authorization",
    }


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DEFAULT_OUT_DIR", "main"]
