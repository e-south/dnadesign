"""TFBS learnability CLI commands for the DenseGen OPAL probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..core.constants import CANDIDATE_RECORDS, DEFAULT_SEED, DENSEGEN_SIDECAR
from ..core.paths import _default_run_root, _repo_root_from, _resolve_repo_path, validate_run_root_policy
from .profiles import resolve_tfbs_target_label_names, tfbs_target_profile_ids
from .retention import (
    DEFAULT_TFBS_STAGE_INITIAL_LABELS,
    DEFAULT_TFBS_STAGE_ROUNDS,
    DEFAULT_TFBS_STAGE_SELECTION_K,
)
from .schema import TFBS_LEARNABILITY_ACTIVE_LABEL_NAMES
from .stage_b.seed import (
    TFBS_STAGE_B_INITIAL_SEED_POLICIES,
    TFBS_STAGE_B_INITIAL_SEED_POLICY_LABEL_VALUE_STRATIFIED_RANDOM,
)
from .stage_b.semantics import TFBS_STAGE_B_DEFAULT_TIE_HANDLING

TFBS_COMMANDS = {
    "tfbs-stage-a",
    "tfbs-stage-b-configs",
    "tfbs-stage-b-prune",
    "tfbs-stage-b-run",
    "tfbs-stage-b-review",
    "tfbs-stage-b-replicate-review",
}


def add_tfbs_subcommands(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register TFBS learnability subcommands on the probe CLI parser."""

    tfbs_stage_a = subparsers.add_parser(
        "tfbs-stage-a",
        help="Materialize DenseGen TFBS learnability Stage A labels, sentinel nulls, and preflight manifests.",
    )
    tfbs_stage_a.add_argument("--candidate-records", default=str(CANDIDATE_RECORDS))
    tfbs_stage_a.add_argument("--densegen-sidecar", default=str(DENSEGEN_SIDECAR))
    tfbs_stage_a.add_argument("--run-root", default=None)
    tfbs_stage_a.add_argument("--run-id", default=None)
    tfbs_stage_a.add_argument("--seed", type=int, default=DEFAULT_SEED)
    tfbs_stage_a.add_argument(
        "--label-name",
        action="append",
        default=[],
        choices=TFBS_LEARNABILITY_ACTIVE_LABEL_NAMES,
        help=(
            "Restrict Stage A null materialization to a TFBS learnability label. Repeatable. "
            "The default is the canonical lexA/cpxR/baeR count_fraction profile."
        ),
    )
    tfbs_stage_a.add_argument(
        "--target-profile",
        default=None,
        choices=tfbs_target_profile_ids(),
        help=(
            "Named TFBS target profile to materialize. Omit for the canonical count_fraction profile. "
            "Cannot be combined with --label-name."
        ),
    )
    tfbs_stage_a.add_argument("--rounds", type=int, default=DEFAULT_TFBS_STAGE_ROUNDS)
    tfbs_stage_a.add_argument("--selection-k", type=int, default=DEFAULT_TFBS_STAGE_SELECTION_K)
    tfbs_stage_a.add_argument("--max-estimated-bytes", type=int, default=None)
    tfbs_stage_a.add_argument(
        "--fail-if-estimate-exceeds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail before writing artifacts when the retention estimate exceeds the configured byte budget.",
    )
    tfbs_stage_a.add_argument(
        "--skip-live-label-rate-sanity",
        action="store_true",
        help="Skip the live-source label-rate sanity check; intended only for fixtures or source snapshots outside v1.",
    )
    tfbs_stage_a.add_argument(
        "--allow-custom-run-root",
        action="store_true",
        help="Allow writes to an external scratch root; repo-local writes stay under .var/studies.",
    )
    tfbs_stage_a.add_argument(
        "--replace-run-root",
        action="store_true",
        help="Delete an existing Stage A run root before materializing artifacts.",
    )
    tfbs_stage_a.add_argument("--json", action="store_true", help="Emit machine-readable JSON summaries.")

    tfbs_stage_b = subparsers.add_parser(
        "tfbs-stage-b-configs",
        help="Generate and optionally validate Dense Array TFBS probe Stage B configs.",
    )
    tfbs_stage_b.add_argument("--stage-a-run-root", required=True)
    tfbs_stage_b.add_argument("--out-dir", default=None)
    tfbs_stage_b.add_argument("--split-id", default="random_id")
    tfbs_stage_b.add_argument("--seed", type=int, default=DEFAULT_SEED)
    tfbs_stage_b.add_argument(
        "--label-name",
        action="append",
        default=[],
        choices=TFBS_LEARNABILITY_ACTIVE_LABEL_NAMES,
        help=(
            "Restrict Stage B config generation to a TFBS learnability label. Repeatable. "
            "The default is the canonical lexA/cpxR/baeR count_fraction profile."
        ),
    )
    tfbs_stage_b.add_argument(
        "--target-profile",
        default=None,
        choices=tfbs_target_profile_ids(),
        help=(
            "Named TFBS target profile to generate. Omit for the canonical count_fraction profile. "
            "Cannot be combined with --label-name."
        ),
    )
    tfbs_stage_b.add_argument("--rounds", type=int, default=DEFAULT_TFBS_STAGE_ROUNDS)
    tfbs_stage_b.add_argument("--selection-k", type=int, default=DEFAULT_TFBS_STAGE_SELECTION_K)
    tfbs_stage_b.add_argument("--initial-label-count", type=int, default=DEFAULT_TFBS_STAGE_INITIAL_LABELS)
    tfbs_stage_b.add_argument(
        "--initial-seed-policy",
        default=TFBS_STAGE_B_INITIAL_SEED_POLICY_LABEL_VALUE_STRATIFIED_RANDOM,
        choices=TFBS_STAGE_B_INITIAL_SEED_POLICIES,
        help=(
            "Initial-label selection policy for synthetic probe campaigns. The default samples randomly within "
            "label-value strata and does not reuse stress batch-0 priors."
        ),
    )
    tfbs_stage_b.add_argument(
        "--tie-handling",
        default=TFBS_STAGE_B_DEFAULT_TIE_HANDLING,
        choices=("competition_rank", "dense_rank", "ordinal"),
        help="Selection tie policy. The default ordinal mode enforces exact top-k acquisition.",
    )
    tfbs_stage_b.add_argument("--score-batch-size", type=int, default=1000)
    tfbs_stage_b.add_argument("--max-x-matrix-gib", type=float, default=8.0)
    tfbs_stage_b.add_argument(
        "--validate-configs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run `uv run opal validate` for each generated probe config.",
    )
    tfbs_stage_b.add_argument(
        "--replace-out-dir",
        action="store_true",
        help="Delete an existing Stage B config output directory before writing.",
    )
    tfbs_stage_b.add_argument(
        "--refresh-existing-execution-state",
        action="store_true",
        help="Refresh generated configs/manifests while preserving existing Stage B campaign outputs.",
    )
    tfbs_stage_b.add_argument("--json", action="store_true", help="Emit machine-readable JSON summaries.")

    tfbs_stage_b_run = subparsers.add_parser(
        "tfbs-stage-b-run",
        help="Run validated Dense Array TFBS probe Stage B campaigns.",
    )
    tfbs_stage_b_run.add_argument("--config-manifest", required=True)
    tfbs_stage_b_run.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Optional round count override for smoke execution; defaults to the manifest round count.",
    )
    tfbs_stage_b_run.add_argument(
        "--campaign-key",
        action="append",
        default=[],
        help="Restrict execution to one campaign key; repeatable. Default runs all manifest campaigns.",
    )
    tfbs_stage_b_run.add_argument(
        "--resume-existing",
        action="store_true",
        help="Allow reuse of existing state and pass OPAL --resume for missing round reruns.",
    )
    tfbs_stage_b_run.add_argument("--json", action="store_true", help="Emit machine-readable JSON summaries.")

    tfbs_stage_b_prune = subparsers.add_parser(
        "tfbs-stage-b-prune",
        help="Hard-delete confounded DenseGen TFBS Stage B campaigns and rewrite generated review manifests.",
    )
    tfbs_stage_b_prune.add_argument("--config-manifest", required=True)
    tfbs_stage_b_prune.add_argument(
        "--label-name",
        action="append",
        required=True,
        help="Stage B label to prune. Repeatable.",
    )
    tfbs_stage_b_prune.add_argument(
        "--delete-review-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Delete generated review and notebook artifacts so they must be regenerated from retained campaigns.",
    )
    tfbs_stage_b_prune.add_argument("--json", action="store_true", help="Emit machine-readable JSON summaries.")

    tfbs_stage_b_review = subparsers.add_parser(
        "tfbs-stage-b-review",
        help="Write realized metadata-enrichment review artifacts for Stage B probe campaigns.",
    )
    tfbs_stage_b_review.add_argument("--config-manifest", required=True)
    tfbs_stage_b_review.add_argument("--out-dir", default=None)
    tfbs_stage_b_review.add_argument(
        "--collection-visual-index",
        default=None,
        help=(
            "Optional OPAL collection_visual_manifest.json to update with realized-label review plots. "
            "When omitted, the Stage B notebook index is auto-detected when present."
        ),
    )
    tfbs_stage_b_review.add_argument("--json", action="store_true", help="Emit machine-readable JSON summaries.")

    tfbs_stage_b_replicate_review = subparsers.add_parser(
        "tfbs-stage-b-replicate-review",
        help=(
            "Write replicated true-label lift, endpoint, claim, and plot artifacts for deterministic "
            "count_fraction Stage B seed pairs."
        ),
    )
    tfbs_stage_b_replicate_review.add_argument(
        "--config-manifest",
        action="append",
        required=True,
        help="Stage B config manifest for one deterministic replicate seed. Repeat exactly for seeds 7, 17, and 29.",
    )
    tfbs_stage_b_replicate_review.add_argument("--out-dir", required=True)
    tfbs_stage_b_replicate_review.add_argument(
        "--collection-visual-index",
        default=None,
        help="Optional OPAL collection_visual_manifest.json to update with replicated review plots.",
    )
    tfbs_stage_b_replicate_review.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON summaries.",
    )


def handle_tfbs_command(args: argparse.Namespace) -> int | None:
    """Dispatch TFBS learnability commands, or return None for non-TFBS commands."""

    if args.command == "tfbs-stage-a":
        return _tfbs_stage_a(args)
    if args.command == "tfbs-stage-b-configs":
        return _tfbs_stage_b_configs(args)
    if args.command == "tfbs-stage-b-run":
        return _tfbs_stage_b_run(args)
    if args.command == "tfbs-stage-b-prune":
        return _tfbs_stage_b_prune(args)
    if args.command == "tfbs-stage-b-review":
        return _tfbs_stage_b_review(args)
    if args.command == "tfbs-stage-b-replicate-review":
        return _tfbs_stage_b_replicate_review(args)
    return None


def _resolve_cli_target_label_names(args: argparse.Namespace) -> tuple[str, ...]:
    return resolve_tfbs_target_label_names(
        target_profile_id=getattr(args, "target_profile", None),
        label_names=tuple(getattr(args, "label_name", ()) or ()),
    )


def _tfbs_stage_a(args: argparse.Namespace) -> int:
    from .retention import DEFAULT_RETENTION_MAX_ESTIMATED_BYTES
    from .stage_a.materialization import TfbsStageAConfig, materialize_tfbs_stage_a

    repo_root = _repo_root_from(Path.cwd())
    run_id = args.run_id or f"densegen_tfbs_learnability_stage_a_seed{args.seed}"
    run_root = (
        _resolve_repo_path(repo_root, Path(args.run_root)) if args.run_root else _default_run_root(repo_root, run_id)
    )
    validate_run_root_policy(repo_root=repo_root, run_root=run_root, allow_custom=bool(args.allow_custom_run_root))
    candidate_records = _resolve_repo_path(repo_root, Path(args.candidate_records))
    densegen_sidecar = None
    if args.densegen_sidecar:
        densegen_sidecar = _resolve_repo_path(repo_root, Path(args.densegen_sidecar))
    max_estimated_bytes = (
        DEFAULT_RETENTION_MAX_ESTIMATED_BYTES if args.max_estimated_bytes is None else int(args.max_estimated_bytes)
    )
    result = materialize_tfbs_stage_a(
        TfbsStageAConfig(
            candidate_records_path=candidate_records,
            densegen_sidecar_path=densegen_sidecar,
            run_root=run_root,
            seed=int(args.seed),
            rounds=int(args.rounds),
            selection_k=int(args.selection_k),
            max_estimated_bytes=max_estimated_bytes,
            fail_if_estimate_exceeds=bool(args.fail_if_estimate_exceeds),
            enforce_live_label_rate_sanity=not bool(args.skip_live_label_rate_sanity),
            replace_run_root=bool(args.replace_run_root),
            label_names=_resolve_cli_target_label_names(args),
            target_profile_id=getattr(args, "target_profile", None),
        )
    )
    payload = result.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("densegen_tfbs_learnability_stage_a")
        print(f"status={payload['status']}")
        print(f"run_root={payload['run_root']}")
        print(f"positive_label_table={payload['positive_label_table_path']}")
        print(f"stage_a_manifest={payload['stage_a_manifest_path']}")
        print(f"retention_estimate={payload['retention_estimate_path']}")
        print(f"null_artifact_count={payload['null_artifact_count']}")
    return 0


def _tfbs_stage_b_configs(args: argparse.Namespace) -> int:
    from .stage_b.configs import TfbsStageBConfig, materialize_tfbs_stage_b_sentinel_configs

    repo_root = _repo_root_from(Path.cwd())
    stage_a_run_root = _resolve_repo_path(repo_root, Path(args.stage_a_run_root))
    out_dir = _resolve_repo_path(repo_root, Path(args.out_dir)) if args.out_dir else None
    result = materialize_tfbs_stage_b_sentinel_configs(
        TfbsStageBConfig(
            stage_a_run_root=stage_a_run_root,
            out_dir=out_dir,
            repo_root=repo_root,
            label_names=_resolve_cli_target_label_names(args),
            target_profile_id=getattr(args, "target_profile", None),
            split_id=str(args.split_id),
            seed=int(args.seed),
            rounds=int(args.rounds),
            selection_k=int(args.selection_k),
            initial_label_count=int(args.initial_label_count),
            initial_seed_policy=str(args.initial_seed_policy),
            selection_tie_handling=str(args.tie_handling),
            validate_configs=bool(args.validate_configs),
            replace_out_dir=bool(args.replace_out_dir),
            refresh_existing_execution_state=bool(args.refresh_existing_execution_state),
            score_batch_size=int(args.score_batch_size),
            max_x_matrix_gib=float(args.max_x_matrix_gib),
        )
    )
    payload = result.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("densegen_tfbs_learnability_stage_b_sentinel_configs")
        print(f"status={payload['status']}")
        print(f"out_dir={payload['out_dir']}")
        print(f"campaign_count={payload['campaign_count']}")
        print(f"validation_status={payload['validation_status']}")
        print(f"config_manifest={payload['config_manifest_path']}")
    return 0


def _tfbs_stage_b_run(args: argparse.Namespace) -> int:
    from .stage_b.execution import TfbsStageBExecutionConfig, run_tfbs_stage_b_sentinel_campaigns

    repo_root = _repo_root_from(Path.cwd())
    config_manifest_path = _resolve_repo_path(repo_root, Path(args.config_manifest))
    result = run_tfbs_stage_b_sentinel_campaigns(
        TfbsStageBExecutionConfig(
            config_manifest_path=config_manifest_path,
            repo_root=repo_root,
            rounds=args.rounds,
            campaign_keys=tuple(args.campaign_key or ()),
            resume_existing=bool(args.resume_existing),
            machine_readable=bool(args.json),
        )
    )
    payload = result.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("densegen_tfbs_learnability_stage_b_sentinel_execution")
        print(f"status={payload['status']}")
        print(f"campaign_count={payload['campaign_count']}")
        print(f"round_count={payload['round_count']}")
        print(f"execution_manifest={payload['execution_manifest_path']}")
    return 0


def _tfbs_stage_b_prune(args: argparse.Namespace) -> int:
    from .stage_b.prune import prune_tfbs_stage_b_campaigns

    repo_root = _repo_root_from(Path.cwd())
    config_manifest_path = _resolve_repo_path(repo_root, Path(args.config_manifest))
    result = prune_tfbs_stage_b_campaigns(
        config_manifest_path,
        prune_label_names=tuple(args.label_name or ()),
        delete_review_artifacts=bool(args.delete_review_artifacts),
    )
    payload = result.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("densegen_tfbs_learnability_stage_b_prune")
        print(f"status={payload['status']}")
        print(f"pruned_campaign_count={payload['pruned_campaign_count']}")
        print(f"retained_campaign_count={payload['retained_campaign_count']}")
        print(f"prune_manifest={payload['prune_manifest_path']}")
    return 0 if payload["status"] == "PASS" else 1


def _tfbs_stage_b_review(args: argparse.Namespace) -> int:
    from .stage_b.review.materialization import build_tfbs_stage_b_realized_label_review

    repo_root = _repo_root_from(Path.cwd())
    config_manifest_path = _resolve_repo_path(repo_root, Path(args.config_manifest))
    out_dir = _resolve_repo_path(repo_root, Path(args.out_dir)) if args.out_dir else None
    collection_visual_index_path = (
        _resolve_repo_path(repo_root, Path(args.collection_visual_index)) if args.collection_visual_index else None
    )
    result = build_tfbs_stage_b_realized_label_review(
        config_manifest_path,
        out_dir=out_dir,
        collection_visual_index_path=collection_visual_index_path,
    )
    payload = result.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("densegen_tfbs_learnability_stage_b_realized_review")
        print(f"status={payload['status']}")
        print(f"review_dir={payload['review_dir']}")
        print(f"trajectory_csv={payload['trajectory_csv_path']}")
        print(f"pair_summary_csv={payload['pair_summary_csv_path']}")
        print(f"claim_assessment_csv={payload['claim_assessment_csv_path']}")
        print(f"plot_manifest_json={payload['plot_manifest_json_path']}")
        print(f"summary_json={payload['summary_json_path']}")
    return 0 if payload["status"] == "PASS" else 1


def _tfbs_stage_b_replicate_review(args: argparse.Namespace) -> int:
    from .stage_b.review.replicates.materialization import build_tfbs_stage_b_replicated_realized_label_review

    repo_root = _repo_root_from(Path.cwd())
    config_manifest_paths = tuple(_resolve_repo_path(repo_root, Path(path)) for path in args.config_manifest)
    out_dir = _resolve_repo_path(repo_root, Path(args.out_dir))
    collection_visual_index_path = (
        _resolve_repo_path(repo_root, Path(args.collection_visual_index)) if args.collection_visual_index else None
    )
    result = build_tfbs_stage_b_replicated_realized_label_review(
        config_manifest_paths,
        out_dir=out_dir,
        collection_visual_index_path=collection_visual_index_path,
    )
    payload = result.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("densegen_tfbs_learnability_stage_b_replicated_review")
        print(f"status={payload['status']}")
        print(f"replicate_seeds={payload['replicate_seeds']}")
        print(f"review_dir={payload['review_dir']}")
        print(f"trajectory_csv={payload['trajectory_csv_path']}")
        print(f"replicate_pair_summary_csv={payload['replicate_pair_summary_csv_path']}")
        print(f"endpoint_summary_csv={payload['endpoint_summary_csv_path']}")
        print(f"claim_assessment_csv={payload['claim_assessment_csv_path']}")
        print(f"plot_manifest_json={payload['plot_manifest_json_path']}")
        print(f"summary_json={payload['summary_json_path']}")
    return 0 if payload["status"] == "PASS" else 1
