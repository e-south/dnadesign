"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/biohub_esmc_wt_mutation_scoring/cli.py

CLI for Eco1 WT-only ESMC masked-marginal mutation scoring.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .constants import (
    DEFAULT_BIOHUB_API_BASE_URL,
    DEFAULT_KEY_FILE,
    DEFAULT_MODEL,
    DEFAULT_POSITIONS,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
)
from .pipeline import materialize_biohub_esmc_wt_mutation_scoring


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Materialize Eco1 WT-only Biohub ESMC masked-marginal mutation scoring artifacts."
    )
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--positions", default=DEFAULT_POSITIONS, help="One-based positions/ranges or 'all'.")
    parser.add_argument("--biohub-api-base-url", default=DEFAULT_BIOHUB_API_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--key-file", type=Path, default=DEFAULT_KEY_FILE)
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Reuse accepted per-position rows whose query hash matches the current request.",
    )
    parser.add_argument(
        "--max-new-requests",
        type=int,
        default=None,
        help="Optional cap on new masked-position logits calls; unattempted rows are explicit.",
    )
    parser.add_argument(
        "--request-sleep-seconds",
        type=float,
        default=0.0,
        help="Delay between new masked-position Biohub logits calls.",
    )
    parser.add_argument("--request-timeout-seconds", type=float, default=DEFAULT_REQUEST_TIMEOUT_SECONDS)
    args = parser.parse_args(argv)

    result = materialize_biohub_esmc_wt_mutation_scoring(
        repo_root=args.repo_root,
        output_root=args.output_root,
        positions=args.positions,
        biohub_api_base_url=args.biohub_api_base_url,
        model=args.model,
        key_file=args.key_file,
        resume_existing=args.resume_existing,
        max_new_requests=args.max_new_requests,
        request_sleep_seconds=args.request_sleep_seconds,
        request_timeout_seconds=args.request_timeout_seconds,
    )
    print(f"selected_positions: {result.selected_position_count}")
    print(f"biohub_request_hash: {result.biohub_request_hash}")
    print(f"position_entropy: {result.position_entropy_path}")
    print(f"substitution_llr: {result.substitution_llr_path}")
    print(f"mask_join: {result.mask_join_path}")
    print(f"plots_root: {result.plots_root}")
    print(f"request_manifest: {result.request_manifest_path}")
    return 0
