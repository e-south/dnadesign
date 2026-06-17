"""Materialize TFBS learning-loop baseline review artifacts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from dnadesign.opal import frozen_round0_scores

from ...stage_a.manifests import file_sha256
from .contracts import (
    COUNT_FIXED_SLOT_POSITION_LEARNING_LOOP_SPEC,
    COUNT_FRACTION_LEARNING_LOOP_SPEC,
    LEARNING_LOOP_BASELINE_SCHEMA_VERSION,
    LEARNING_LOOP_BASELINE_SURFACE_KIND,
    FrozenReplayResult,
    LearningLoopBaselineSpec,
)
from .frames import claim_interpretation_frame, cumulative_lift_trajectory, endpoint_summary_frame
from .plots import materialize_frozen_replay_plots
from .replay import frozen_rank_chunks, top_budget_chunks
from .sources import (
    active_selection_frame,
    campaign_label_table,
    campaign_rows,
    initial_seed_ids,
    load_learning_loop_manifests,
    pair_rows,
    validate_shared_pair_contracts,
)


def build_count_fraction_frozen_round0_replay(
    config_manifest_paths: Iterable[str | Path],
    *,
    out_dir: str | Path,
) -> FrozenReplayResult:
    """Write learning-loop baseline artifacts for replicated count-fraction campaigns."""

    return build_learning_loop_baseline_review(
        config_manifest_paths,
        out_dir=out_dir,
        spec=COUNT_FRACTION_LEARNING_LOOP_SPEC,
    )


def build_count_fixed_slot_position_frozen_round0_replay(
    config_manifest_paths: Iterable[str | Path],
    *,
    out_dir: str | Path,
) -> FrozenReplayResult:
    """Write learning-loop baseline artifacts for count-fixed slot-position campaigns."""

    return build_learning_loop_baseline_review(
        config_manifest_paths,
        out_dir=out_dir,
        spec=COUNT_FIXED_SLOT_POSITION_LEARNING_LOOP_SPEC,
    )


def build_learning_loop_baseline_review(
    config_manifest_paths: Iterable[str | Path],
    *,
    out_dir: str | Path,
    spec: LearningLoopBaselineSpec,
) -> FrozenReplayResult:
    """Write active/frozen/top-budget baseline artifacts for completed TFBS Stage B campaigns."""

    manifests = load_learning_loop_manifests([Path(path) for path in config_manifest_paths], spec=spec)
    for manifest in manifests:
        validate_shared_pair_contracts(manifest)
    review_dir = Path(out_dir)
    review_dir.mkdir(parents=True, exist_ok=True)
    rounds = _single_int(manifests, "rounds")
    selection_k = _single_int(manifests, "selection_k")
    all_campaigns = [campaign for manifest in manifests for campaign in campaign_rows(manifest)]
    all_pairs = [_pair_dict(pair) for manifest in manifests for pair in pair_rows(manifest)]

    trajectories: list[pd.DataFrame] = []
    for campaign in all_campaigns:
        trajectories.append(_active_trajectory(campaign, rounds=rounds, selection_k=selection_k))
        trajectories.append(_frozen_trajectory(campaign, rounds=rounds, selection_k=selection_k))
        trajectories.append(_top_budget_trajectory(campaign, rounds=rounds, selection_k=selection_k))
    trajectory = pd.concat(trajectories, ignore_index=True)
    endpoint_summary = endpoint_summary_frame(trajectory, pairs=all_pairs)
    claim_interpretation = claim_interpretation_frame(endpoint_summary)

    trajectory_path = review_dir / "learning_loop_baseline_trajectory.csv"
    endpoint_path = review_dir / "learning_loop_baseline_endpoint_summary.csv"
    claim_path = review_dir / "learning_loop_baseline_claim_interpretation.csv"
    manifest_path = review_dir / "learning_loop_baseline_manifest.json"
    trajectory.to_csv(trajectory_path, index=False)
    endpoint_summary.to_csv(endpoint_path, index=False)
    claim_interpretation.to_csv(claim_path, index=False)
    plot_manifest_path = materialize_frozen_replay_plots(
        trajectory_csv_path=trajectory_path,
        endpoint_summary_csv_path=endpoint_path,
        claim_interpretation_csv_path=claim_path,
        out_dir=review_dir / "plots",
        spec=spec,
    )
    summary = _manifest_payload(
        spec=spec,
        source_manifests=[Path(path) for path in config_manifest_paths],
        manifests=manifests,
        trajectory_path=trajectory_path,
        endpoint_path=endpoint_path,
        claim_path=claim_path,
        plot_manifest_path=plot_manifest_path,
        trajectory=trajectory,
        claim_interpretation=claim_interpretation,
    )
    _write_json(manifest_path, summary)
    return FrozenReplayResult(
        status=str(summary["status"]),
        review_dir=review_dir,
        manifest_json_path=manifest_path,
        trajectory_csv_path=trajectory_path,
        endpoint_summary_csv_path=endpoint_path,
        claim_interpretation_csv_path=claim_path,
        plot_manifest_json_path=plot_manifest_path,
    )


def _active_trajectory(campaign: Mapping[str, Any], *, rounds: int, selection_k: int) -> pd.DataFrame:
    label_name = str(campaign["label_name"])
    labels = campaign_label_table(campaign)
    pool_baseline = float(labels[label_name].mean())
    selections = active_selection_frame(campaign, rounds=rounds)
    return cumulative_lift_trajectory(
        selections,
        labels,
        label_name=label_name,
        pool_baseline=pool_baseline,
        campaign_key=str(campaign["campaign_key"]),
        oracle_role=str(campaign["oracle_role"]),
        scientific_control_role=_scientific_control_role(campaign),
        seed=int(campaign["seed"]),
        selection_k=selection_k,
    )


def _frozen_trajectory(campaign: Mapping[str, Any], *, rounds: int, selection_k: int) -> pd.DataFrame:
    label_name = str(campaign["label_name"])
    labels = campaign_label_table(campaign)
    pool_baseline = float(labels[label_name].mean())
    scores, seed_ids = frozen_round0_scores(Path(str(campaign["config_path"])))
    selections = frozen_rank_chunks(scores, selection_k=selection_k, rounds=rounds, excluded_ids=set(seed_ids))
    return cumulative_lift_trajectory(
        selections,
        labels,
        label_name=label_name,
        pool_baseline=pool_baseline,
        campaign_key=str(campaign["campaign_key"]),
        oracle_role=str(campaign["oracle_role"]),
        scientific_control_role=_scientific_control_role(campaign),
        seed=int(campaign["seed"]),
        selection_k=selection_k,
    )


def _top_budget_trajectory(campaign: Mapping[str, Any], *, rounds: int, selection_k: int) -> pd.DataFrame:
    label_name = str(campaign["label_name"])
    labels = campaign_label_table(campaign)
    pool_baseline = float(labels[label_name].mean())
    seed_ids = initial_seed_ids(Path(str(campaign["initial_label_input_path"])))
    selections = top_budget_chunks(
        labels,
        label_name=label_name,
        selection_k=selection_k,
        rounds=rounds,
        excluded_ids=set(seed_ids),
    )
    return cumulative_lift_trajectory(
        selections,
        labels,
        label_name=label_name,
        pool_baseline=pool_baseline,
        campaign_key=str(campaign["campaign_key"]),
        oracle_role=str(campaign["oracle_role"]),
        scientific_control_role=_scientific_control_role(campaign),
        seed=int(campaign["seed"]),
        selection_k=selection_k,
    )


def _manifest_payload(
    *,
    spec: LearningLoopBaselineSpec,
    source_manifests: list[Path],
    manifests: list[Mapping[str, Any]],
    trajectory_path: Path,
    endpoint_path: Path,
    claim_path: Path,
    plot_manifest_path: Path,
    trajectory: pd.DataFrame,
    claim_interpretation: pd.DataFrame,
) -> dict[str, Any]:
    labels = sorted(trajectory["label_name"].astype(str).unique().tolist())
    seeds = sorted(int(value) for value in trajectory["seed"].unique().tolist())
    source_profiles = _source_profiles(manifests)
    adaptive_supported = int(
        (claim_interpretation["interpretation_status"].astype(str) == "ADAPTIVE_GAIN_SUPPORTED").sum()
    )
    return {
        "schema_version": LEARNING_LOOP_BASELINE_SCHEMA_VERSION,
        "status": "PASS",
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "surface_kind": LEARNING_LOOP_BASELINE_SURFACE_KIND,
        "review_id": spec.review_id,
        "comparison_set_key": spec.comparison_set_key,
        "comparison_set_label": spec.comparison_set_label,
        "visual_tier": spec.visual_tier,
        "source_config_manifest_paths": [str(path) for path in source_manifests],
        "source_config_manifest_hashes": [file_sha256(path) for path in source_manifests],
        "source_profile": dict(source_profiles[0]) if len(source_profiles) == 1 else None,
        "source_profiles": source_profiles,
        "source_profile_ids": [str(profile["profile_id"]) for profile in source_profiles],
        "replicate_seeds": seeds,
        "replicate_count": len(seeds),
        "label_names": labels,
        "campaign_count": int(trajectory["campaign_key"].nunique()),
        "pair_count": int(sum(len(pair_rows(manifest)) for manifest in manifests)),
        "rounds": int(_single_int(manifests, "rounds")),
        "selection_k": int(_single_int(manifests, "selection_k")),
        "trajectory_csv_path": str(trajectory_path),
        "trajectory_csv_hash": file_sha256(trajectory_path),
        "endpoint_summary_csv_path": str(endpoint_path),
        "endpoint_summary_csv_hash": file_sha256(endpoint_path),
        "claim_interpretation_csv_path": str(claim_path),
        "claim_interpretation_csv_hash": file_sha256(claim_path),
        "plot_manifest_json_path": str(plot_manifest_path),
        "plot_manifest_json_hash": file_sha256(plot_manifest_path),
        "adaptive_gain_supported_label_count": adaptive_supported,
        "claim_boundary": spec.claim_boundary,
        "interpretation_boundary": spec.interpretation_boundary,
    }


def _single_int(manifests: list[Mapping[str, Any]], field: str) -> int:
    values = sorted({int(manifest[field]) for manifest in manifests})
    if len(values) != 1:
        raise ValueError(f"Frozen replay manifests disagree on {field}: {values}")
    return values[0]


def _scientific_control_role(campaign: Mapping[str, Any]) -> str:
    if str(campaign["oracle_role"]) == "positive":
        return ""
    return str(campaign.get("null_control_role") or "")


def _pair_dict(pair: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "label_name": str(pair["label_name"]),
        "seed": int(pair["seed"]),
        "positive_campaign_key": str(pair["positive_campaign_key"]),
        "null_campaign_key": str(pair["null_campaign_key"]),
    }


def _source_profiles(manifests: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    for manifest in manifests:
        profile = dict(manifest["target_profile"])
        profiles.setdefault(str(profile["profile_id"]), profile)
    return [profiles[key] for key in sorted(profiles)]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
