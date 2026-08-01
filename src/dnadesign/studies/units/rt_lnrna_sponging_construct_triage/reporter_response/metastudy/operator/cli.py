"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reporter_response/metastudy/operator/cli.py

Command-line adapter for canonical meta-study regeneration and verification.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from ..contracts.decision import decision_is_evidence_bearing, decision_to_dict
from ..publication import publish_metastudy, verify_publication
from ..sensitivity import sensitivity_evaluations_to_payload
from .checkout import require_active_dnadesign_checkout
from .persistence import write_source_controlled_state
from .regeneration import regenerate_metastudy, validate_live_source_controlled_state
from .state import STATE_FILE


def build_parser() -> argparse.ArgumentParser:
    """Build the stable command contract without executing study I/O."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    regenerate = subparsers.add_parser(
        "regenerate",
        help="Reconstruct and optionally publish one canonical meta-study generation",
    )
    regenerate.add_argument("--phd-root", type=Path, required=True)
    regenerate.add_argument(
        "--dnadesign-root",
        type=Path,
        help="Active Dnadesign source checkout. Defaults to the checkout running this command.",
    )
    regenerate.add_argument("--publication", type=Path)
    regenerate.add_argument("--state-dir", type=Path)
    status = subparsers.add_parser("status", help="Validate and summarize one source-controlled state generation")
    status.add_argument("--phd-root", type=Path, required=True)
    status.add_argument(
        "--dnadesign-root",
        type=Path,
        help="Active Dnadesign source checkout. Defaults to the checkout running this command.",
    )
    status.add_argument("--state-dir", type=Path, required=True)
    verify = subparsers.add_parser("verify", help="Verify one create-only meta-study publication")
    verify.add_argument("--publication", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one fail-closed meta-study operator command."""

    args = build_parser().parse_args(argv)
    if args.command == "verify":
        verify_publication(args.publication)
        print(json.dumps({"ok": True, "publication": str(args.publication.resolve())}, sort_keys=True))
        return 0
    dnadesign_root = require_active_dnadesign_checkout(args.dnadesign_root)
    if args.command == "status":
        validation = validate_live_source_controlled_state(
            args.state_dir / STATE_FILE,
            phd_root=args.phd_root,
            dnadesign_root=dnadesign_root,
        )
        state = validation.state
        decision = state["decision"]
        assert isinstance(decision, dict)
        print(
            json.dumps(
                {
                    "generation_digest": state["generation_digest"],
                    "status": decision["status"],
                    "selected_reduction": decision["selected_reduction"],
                    "blockers": decision["blockers"],
                    "limitations": decision["limitations"],
                    "objective_readiness": state["objective_readiness"],
                    "sensitivity_evaluations": state["sensitivity_evaluations"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    result = regenerate_metastudy(
        phd_root=args.phd_root,
        dnadesign_root=dnadesign_root,
    )
    state_paths = None
    if args.state_dir is not None:
        state_paths = write_source_controlled_state(
            result,
            destination=args.state_dir,
            phd_root=args.phd_root,
        )
    publication = None
    if args.publication is not None:
        publication = publish_metastudy(
            result.decision,
            args.publication,
            primary_evidence=(result.primary_evidence if decision_is_evidence_bearing(result.decision) else ()),
            sensitivity_evidence=result.sensitivity_evidence,
            sensitivity_evaluations=result.sensitivity_evaluations,
            sensitivity_coverages=result.sensitivity_coverages,
            objective_readiness=result.objective_readiness,
        )
    payload = decision_to_dict(result.decision)
    payload["decision_digest"] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    )
    payload["publication"] = str(publication) if publication is not None else None
    payload["objective_readiness"] = asdict(result.objective_readiness)
    payload["sensitivity_evaluations"] = sensitivity_evaluations_to_payload(result.sensitivity_evaluations)
    payload["state_paths"] = [str(path) for path in state_paths] if state_paths is not None else None
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0
