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
from .runtime.calibration_preview import preview_response_calibration

DEFAULT_OUT_DIR = Path(
    "src/dnadesign/studies/units/stress_ethanol_cipro_growth/workbench/outputs/response_metastudy/latest"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate response-label, objective, model, and selection alternatives for the stress study."
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--reader-root",
        type=Path,
        required=True,
        help="Reader repository root providing the public reader CLI.",
    )
    parser.add_argument(
        "--reader-experiment",
        type=Path,
        required=True,
        help="Canonical response-window output experiment with a verified RecordStore catalog.",
    )
    parser.add_argument(
        "--candidate-bindings",
        type=Path,
        required=True,
        help="Verified stress-study promoter candidate-binding bundle.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--calibration-preview",
        action="store_true",
        help="Derive Reader-backed RMF calibration without writing metastudy output.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")
    repo_root = args.repo_root.resolve()
    if args.calibration_preview:
        if args.overwrite:
            parser.error("--calibration-preview is read-only and cannot be combined with --overwrite.")
        preview = preview_response_calibration(
            repo_root=repo_root,
            reader_root=args.reader_root.resolve(),
            reader_experiment_root=args.reader_experiment.resolve(),
            candidate_binding_bundle_root=args.candidate_bindings.resolve(),
        )
        if args.json:
            print(json.dumps(preview, allow_nan=False, indent=2, sort_keys=True))
        else:
            print("stress_ethanol_cipro_growth RMF calibration preview")
            print(f"primary_reduction={preview['primary_reduction_id']}")
            print(f"ready_for_campaign={preview['ready_for_campaign']}")
            print(f"source_ready={preview['source_ready']}")
            print(f"reader_record_receipt_sha256={preview['reader_record_receipt_sha256']}")
            print(f"campaign_matches={preview['campaign_matches_reader_calibration']}")
            for blocker in preview["blockers"]:  # type: ignore[union-attr]
                print(f"blocker={blocker}")
            for row in preview["selection_views"]:  # type: ignore[union-attr]
                print(f"{row['selection_view_id']}={row['derived_calibration']}")
        return 0
    out_dir = args.out_dir
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    manifest = run_metastudy(
        repo_root=repo_root,
        reader_root=args.reader_root.resolve(),
        reader_experiment_root=args.reader_experiment.resolve(),
        candidate_binding_bundle_root=args.candidate_bindings.resolve(),
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
