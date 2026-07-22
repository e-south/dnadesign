"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/foldcheck_report/cli.py

CLI for Eco1 fold-check report materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.foldcheck_report.pipeline import (
    materialize_foldcheck_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize Eco1 foldcheck_report.parquet from ColabFold outputs.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--source-output-root", type=Path, default=None)
    parser.add_argument("--colabfold-output-root", type=Path, required=True)
    parser.add_argument("--runtime-version", required=True)
    parser.add_argument("--runtime-parameter", action="append", default=[], metavar="KEY=VALUE")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = materialize_foldcheck_report(
        repo_root=args.repo_root,
        output_root=args.output_root,
        source_output_root=args.source_output_root,
        colabfold_output_root=args.colabfold_output_root,
        runtime_version=args.runtime_version,
        runtime_parameters=_parse_runtime_parameters(args.runtime_parameter),
    )
    print(result.foldcheck_report_path)
    return 0


def _parse_runtime_parameters(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"runtime parameter must be KEY=VALUE: {value!r}")
        key, raw = value.split("=", 1)
        if not key.strip():
            raise ValueError(f"runtime parameter key cannot be empty: {value!r}")
        parsed[key.strip()] = raw.strip()
    return parsed
