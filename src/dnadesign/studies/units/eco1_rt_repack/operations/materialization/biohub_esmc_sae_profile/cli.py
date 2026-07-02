"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/biohub_esmc_sae_profile/cli.py

CLI for Eco1 Biohub ESMC SAE-profile materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.biohub_esmc_sae_profile.constants import (
    DEFAULT_BIOHUB_API_BASE_URL,
    DEFAULT_KEY_FILE,
    DEFAULT_MODEL,
    DEFAULT_NORMALIZE_FEATURES,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_SAE_MODEL,
    DEFAULT_SEQUENCE_LIMIT,
)

from .feature_description_enrichment import (
    enrich_existing_biohub_esmc_feature_catalog,
)
from .pipeline import (
    materialize_biohub_esmc_sae_profile,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Materialize Eco1 Biohub ESMC SAE-profile artifacts.")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--feature-descriptions-only",
        action="store_true",
        help="Enrich only the existing feature catalog; do not rebuild profile or sparse feature tables.",
    )
    parser.add_argument("--sequence-limit", default=DEFAULT_SEQUENCE_LIMIT)
    parser.add_argument("--biohub-api-base-url", default=DEFAULT_BIOHUB_API_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sae-model", default=DEFAULT_SAE_MODEL)
    parser.add_argument(
        "--normalize-features",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_NORMALIZE_FEATURES,
        help="Request Biohub normalized SAE features when supported by the selected SAE model.",
    )
    parser.add_argument("--key-file", type=Path, default=DEFAULT_KEY_FILE)
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Reuse accepted rows whose per-sequence Biohub query hash matches the current request.",
    )
    parser.add_argument(
        "--max-new-requests",
        type=int,
        default=None,
        help="Optional cap on new Biohub API calls; unattempted rows are explicit and resumable.",
    )
    parser.add_argument(
        "--request-sleep-seconds",
        type=float,
        default=0.0,
        help="Delay between new Biohub API calls for conservative batch progression.",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        help="Wall-clock timeout for each Biohub POST request.",
    )
    parser.add_argument(
        "--fetch-feature-descriptions",
        action="store_true",
        help=(
            "Fetch public Biohub feature descriptions for compatible SAE dictionaries. "
            "The Eco1 happy path uses the described 6B/layer60/16k dictionary."
        ),
    )
    parser.add_argument(
        "--feature-description-limit",
        type=int,
        default=None,
        help="Optional cap on feature-description GET requests when enrichment is explicitly enabled.",
    )
    parser.add_argument(
        "--feature-description-batch-size",
        type=int,
        default=100,
        help="Checkpoint feature-description enrichment after this many GET requests.",
    )
    parser.add_argument(
        "--feature-description-sleep-seconds",
        type=float,
        default=0.0,
        help="Delay between feature-description GET requests when enrichment is explicitly enabled.",
    )
    args = parser.parse_args(argv)

    if args.feature_descriptions_only:
        result = enrich_existing_biohub_esmc_feature_catalog(
            repo_root=args.repo_root,
            output_root=args.output_root,
            biohub_api_base_url=args.biohub_api_base_url,
            sae_model=args.sae_model,
            request_timeout_seconds=args.request_timeout_seconds,
            feature_description_limit=args.feature_description_limit,
            feature_description_batch_size=args.feature_description_batch_size,
            feature_description_sleep_seconds=args.feature_description_sleep_seconds,
            progress_callback=_print_feature_description_progress,
        )
        print(f"feature_catalog: {result.feature_catalog_path}")
        print(f"feature_description_manifest: {result.manifest_path}")
        print(f"observed_feature_count: {result.observed_feature_count}")
        print(f"enriched_feature_count: {result.enriched_feature_count}")
        return 0

    result = materialize_biohub_esmc_sae_profile(
        repo_root=args.repo_root,
        output_root=args.output_root,
        sequence_limit=args.sequence_limit,
        biohub_api_base_url=args.biohub_api_base_url,
        model=args.model,
        sae_model=args.sae_model,
        normalize_features=args.normalize_features,
        key_file=args.key_file,
        resume_existing=args.resume_existing,
        max_new_requests=args.max_new_requests,
        request_sleep_seconds=args.request_sleep_seconds,
        request_timeout_seconds=args.request_timeout_seconds,
        fetch_feature_descriptions=args.fetch_feature_descriptions,
        feature_description_limit=args.feature_description_limit,
        feature_description_sleep_seconds=args.feature_description_sleep_seconds,
    )
    print(f"selected_sequences: {result.selected_sequence_count}")
    print(f"biohub_request_hash: {result.biohub_request_hash}")
    print(f"profile: {result.profile_path}")
    print(f"protein_features: {result.protein_features_path}")
    print(f"residue_features: {result.residue_features_path}")
    print(f"feature_catalog: {result.feature_catalog_path}")
    print(f"request_manifest: {result.request_manifest_path}")
    return 0


def _print_feature_description_progress(summary: dict[str, object]) -> None:
    print(
        "feature_description_batch: "
        f"batch={summary['batch_count']} "
        f"attempted={summary['cumulative_attempted_feature_count']} "
        f"enriched={summary['enriched_feature_count']} "
        f"missing={summary['missing_feature_description_count']}",
        flush=True,
    )
