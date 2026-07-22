"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/model_evidence/cli.py

Command-line surface for immutable model-evidence trajectories.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .storage import rebuild_catalog, record_checkpoint, verify_trajectory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record and verify study-owned model-evidence trajectory checkpoints.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    record = subparsers.add_parser("record", help="Record a verified metastudy bundle as an immutable checkpoint.")
    record.add_argument("--metastudy-bundle", type=Path, required=True)
    record.add_argument("--trajectory-root", type=Path, required=True)
    record.add_argument("--evidence-id", required=True)
    record.add_argument("--json", action="store_true")

    verify = subparsers.add_parser("verify", help="Verify immutable records and replaceable index references.")
    verify.add_argument("--trajectory-root", type=Path, required=True)
    verify.add_argument("--json", action="store_true")

    rebuild = subparsers.add_parser("rebuild-catalog", help="Rebuild the catalog from immutable checkpoints.")
    rebuild.add_argument("--trajectory-root", type=Path, required=True)
    rebuild.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "record":
        payload = record_checkpoint(
            metastudy_bundle=args.metastudy_bundle,
            trajectory_root=args.trajectory_root,
            evidence_id=args.evidence_id,
        )
    elif args.command == "verify":
        payload = verify_trajectory(args.trajectory_root)
    else:
        payload = rebuild_catalog(args.trajectory_root)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(_plain_summary(args.command, payload))
    return 0


def _plain_summary(command: str, payload: dict[str, object]) -> str:
    if command == "record":
        return "\n".join(
            (
                "model-evidence checkpoint recorded",
                f"evidence_id={payload['evidence_id']}",
                f"protocol_digest={payload['protocol_digest']}",
                f"checkpoint_digest={payload['checkpoint_digest']}",
                f"checkpoint_path={payload['checkpoint_path']}",
            )
        )
    return "\n".join(
        (
            "model-evidence trajectory verified" if command == "verify" else "model-evidence catalog rebuilt",
            f"protocol_count={payload['protocol_count']}",
            f"checkpoint_count={payload['checkpoint_count']}",
        )
    )


__all__ = ["build_parser", "main"]
