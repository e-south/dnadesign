"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/contact_geometry/cli.py

CLI for Eco1 RT contact-geometry profile materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.contact_geometry.constants import (
    _DEFAULT_CREATED_AT,
    _DEFAULT_OUTPUT_ROOT,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.contact_geometry.pipeline import (
    materialize_contact_geometry_profile,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the contact-geometry CLI parser."""

    parser = argparse.ArgumentParser(
        description="Materialize Eco1 RT contact_geometry_profile.parquet from the selected 7V9U protomer model."
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=_DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--created-at", default=_DEFAULT_CREATED_AT)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run contact-geometry materialization and print emitted paths."""

    args = build_parser().parse_args(argv)
    result = materialize_contact_geometry_profile(
        repo_root=args.repo_root,
        output_root=args.output_root,
        created_at=args.created_at,
    )
    print(json.dumps({"contact_geometry_profile_path": str(result.contact_geometry_profile_path)}, indent=2))
    return 0
