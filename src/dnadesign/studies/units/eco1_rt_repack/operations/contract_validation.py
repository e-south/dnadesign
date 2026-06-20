"""Command-line entrypoint for Eco1 RT repack contract validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.constants import _ALLOWED_PHASES
from dnadesign.studies.units.eco1_rt_repack.operations.contracts.suite import validate_checked_in_contracts


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate Eco1 RT repack checked-in contracts.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--phase", choices=_ALLOWED_PHASES, default="phase0_scaffold")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = validate_checked_in_contracts(repo_root=args.repo_root, phase=args.phase, output_root=args.output_root)
    print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
