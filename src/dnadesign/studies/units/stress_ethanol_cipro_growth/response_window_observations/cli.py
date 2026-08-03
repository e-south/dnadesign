"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/response_window_observations/cli.py

Preview, materialize, and verify study-owned response observations.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .artifact import materialize_response_window_observations, verify_response_window_observations
from .censoring import bounded_primary_summary
from .sources import ResponseWindowObservationEvidence, preview_response_window_observation_evidence

CONFIG_ROOT = Path(__file__).resolve().parent / "config"
DEFAULT_POLICY_PATH = CONFIG_ROOT / "observation_policy.yaml"
DEFAULT_READER_PROJECTION_PATH = CONFIG_ROOT / "reader_response_projection.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage stress-study candidate observations from verified Reader response-window evidence."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preview = subparsers.add_parser("preview", help="Validate sources and report label-truth blockers.")
    _add_source_inputs(preview)

    materialize = subparsers.add_parser(
        "materialize",
        help="Publish an approved, blocker-free observation bundle atomically.",
    )
    _add_source_inputs(materialize)
    materialize.add_argument("--out-dir", type=Path, required=True)
    materialize.add_argument("--allowed-output-root", type=Path, required=True)

    verify = subparsers.add_parser("verify", help="Verify a published observation bundle.")
    verify.add_argument("--bundle-dir", type=Path, required=True)
    verify.add_argument("--allowed-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "verify":
        verified = verify_response_window_observations(args.bundle_dir, allowed_root=args.allowed_root)
        payload: dict[str, object] = {
            "manifest_json": str(verified.manifest_json),
            "manifest_sha256": verified.manifest_sha256,
            "candidate_count": verified.candidate_count,
            "policy_id": verified.policy_id,
            "y_space": verified.y_space,
            "verified": True,
        }
    else:
        evidence = preview_response_window_observation_evidence(
            reader_root=args.reader_root,
            reader_experiment_root=args.reader_experiment,
            reader_projection_path=args.reader_projection,
            candidate_bindings_root=args.candidate_bindings,
            policy_path=args.policy,
        )
        if args.command == "preview":
            payload = _preview_payload(evidence)
        else:
            written = materialize_response_window_observations(
                evidence,
                out_dir=args.out_dir,
                allowed_output_root=args.allowed_output_root,
            )
            payload = {
                **_preview_payload(evidence),
                "manifest_json": str(written.manifest_json),
                "observations_parquet": str(written.observations_parquet),
                "materialized": True,
            }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _add_source_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--reader-root", type=Path, required=True)
    parser.add_argument("--reader-experiment", type=Path, required=True)
    parser.add_argument("--candidate-bindings", type=Path, required=True)
    parser.add_argument("--reader-projection", type=Path, default=DEFAULT_READER_PROJECTION_PATH)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)


def _preview_payload(evidence: ResponseWindowObservationEvidence) -> dict[str, object]:
    primary = evidence.resolved.measurements.loc[
        evidence.resolved.measurements["reduction_id"].astype(str).eq(evidence.policy.aggregation.primary_reduction_id)
    ]
    experiment_counts = primary.groupby("candidate_id")["reader_experiment_id"].nunique()
    repeated = set(experiment_counts.loc[experiment_counts.gt(1)].index.astype(str))
    publishable = set(evidence.preview.observations["candidate_id"].astype(str))
    diagnostics = evidence.preview.repeat_diagnostics
    repeat_statuses = (
        diagnostics.loc[:, ["candidate_id", "status"]].drop_duplicates().set_index("candidate_id")["status"].astype(str)
    )
    status_counts = repeat_statuses.value_counts().to_dict()
    maximum_range = None if diagnostics.empty else float(diagnostics["range"].max())
    blockers = list(evidence.preview.blockers)
    return {
        "study_id": "stress_ethanol_cipro_growth",
        "policy_id": evidence.policy.policy_id,
        "approval_status": evidence.policy.approval_status,
        "reader_catalog_sha256": evidence.reader_catalog_sha256,
        "reader_projection_sha256": evidence.reader_projection_sha256,
        "reader_record_receipt_sha256": evidence.reader_record_receipt_sha256,
        "candidate_bindings_manifest_sha256": evidence.candidate_bindings_manifest_sha256,
        "candidate_count": int(primary["candidate_id"].nunique()),
        "candidate_observation_preview_count": len(publishable),
        "repeated_candidate_count": len(repeated),
        "selected_repeated_candidate_count": int(status_counts.get("label_source_selected", 0)),
        "excluded_candidate_count": int(status_counts.get("label_source_excluded", 0)),
        "remeasure_required_candidate_count": int(status_counts.get("remeasure_required", 0)),
        "blocked_repeated_candidate_count": int(status_counts.get("review_required", 0)),
        "maximum_repeat_component_range": maximum_range,
        "excluded_reader_design_count": len(evidence.resolved.excluded_reader_designs),
        **bounded_primary_summary(evidence.preview.contributions),
        "blocker_count": len(blockers),
        "blockers": blockers,
        "ready_to_materialize": not blockers and evidence.policy.approval_status == "approved",
    }


__all__ = [
    "DEFAULT_POLICY_PATH",
    "DEFAULT_READER_PROJECTION_PATH",
    "build_parser",
    "main",
]
