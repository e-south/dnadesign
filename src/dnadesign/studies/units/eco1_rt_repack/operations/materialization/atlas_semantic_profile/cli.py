"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/atlas_semantic_profile/cli.py

CLI for Eco1 ESM Atlas semantic-profile materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.atlas_semantic_profile.constants import (
    ATLAS_API_BASE_URL,
    DEFAULT_SEQUENCE_LIMIT,
    DEFAULT_TOPK_FEATURES,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.atlas_semantic_profile.pipeline import (
    materialize_atlas_semantic_profile,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Materialize Eco1 ESM Atlas semantic-profile artifacts.")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--sequence-limit", default=DEFAULT_SEQUENCE_LIMIT)
    parser.add_argument("--atlas-api-base-url", default=ATLAS_API_BASE_URL)
    parser.add_argument("--topk-features", type=int, default=DEFAULT_TOPK_FEATURES)
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        default=None,
        help="Optional eco1_rt.atlas_subset_manifest selecting exact fold-accepted sequence ids.",
    )
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Reuse existing rows whose per-sequence Atlas query hash matches the current request.",
    )
    parser.add_argument(
        "--max-new-requests",
        type=int,
        default=None,
        help="Optional cap on new Atlas API calls; unattempted rows are explicit and resumable.",
    )
    parser.add_argument(
        "--request-sleep-seconds",
        type=float,
        default=0.0,
        help="Delay between new Atlas API calls for polite batch progression.",
    )
    parser.add_argument(
        "--prediction-set-id",
        default=None,
        help="Required when --allow-fold-on-miss is used; names Atlas-generated structure predictions.",
    )
    parser.add_argument(
        "--allow-fold-on-miss",
        action="store_true",
        help="Permit Atlas on-demand folding. Default is false for semantic-profile smoke runs.",
    )
    args = parser.parse_args(argv)

    result = materialize_atlas_semantic_profile(
        repo_root=args.repo_root,
        output_root=args.output_root,
        sequence_limit=args.sequence_limit,
        atlas_api_base_url=args.atlas_api_base_url,
        topk_features=args.topk_features,
        allow_fold_on_miss=args.allow_fold_on_miss,
        prediction_set_id=args.prediction_set_id,
        selection_manifest_path=args.selection_manifest,
        resume_existing=args.resume_existing,
        max_new_requests=args.max_new_requests,
        request_sleep_seconds=args.request_sleep_seconds,
    )
    print(f"selected_sequences: {result.selected_sequence_count}")
    print(f"atlas_request_hash: {result.atlas_request_hash}")
    print(f"profile: {result.profile_path}")
    print(f"protein_activations: {result.protein_activations_path}")
    print(f"residue_activations: {result.residue_activations_path}")
    print(f"feature_catalog: {result.feature_catalog_path}")
    print(f"structure_prediction_registry: {result.structure_prediction_registry_path}")
    return 0
