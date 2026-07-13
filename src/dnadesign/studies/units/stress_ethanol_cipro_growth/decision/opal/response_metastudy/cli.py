"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/cli.py

Command-line entrypoint for the stress-study response metric metastudy.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runtime.audit import run_metastudy

DEFAULT_OUT_DIR = Path(
    "src/dnadesign/studies/units/stress_ethanol_cipro_growth/workbench/outputs/response_metastudy/latest"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate response-label, objective, model, and selection alternatives for the stress study."
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--reader-bundle",
        type=Path,
        required=True,
        help="Reader response-window bundle using reader.response_window.bundle.v3.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")
    repo_root = args.repo_root.resolve()
    out_dir = args.out_dir
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    manifest = run_metastudy(
        repo_root=repo_root,
        reader_bundle_root=args.reader_bundle.resolve(),
        out_dir=out_dir,
        overwrite=bool(args.overwrite),
        top_k=int(args.top_k),
    )
    if args.json:
        print(json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True))
    else:
        print("stress_ethanol_cipro_growth response metric metastudy")
        print(f"out_dir={manifest['output_dir']}")
        print(f"verdict={manifest['recommendation']['verdict']}")
        print(f"promoted_policy={manifest['recommendation']['promoted_policy_id'] or 'none'}")
    return 0


__all__ = ["main"]
