"""Data-frame builders for DenseGen TFBS Stage B realized-label review."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ...schema import (
    TFBS_LEARNABILITY_FAMILY_CONTENT_NULL_VERSION,
    TFBS_LEARNABILITY_SLOT_GEOMETRY_NULL_VERSION,
)
from .contracts import VALID_NEGATIVE_CONTROL
from .io import campaign_workdir, label_table, selection_table


def trajectory_frame(campaigns: Sequence[Mapping[str, Any]], *, rounds: int) -> pd.DataFrame:
    """Build per-round realized selected-label trajectory rows."""

    rows: list[dict[str, Any]] = []
    if rounds <= 0:
        raise ValueError("Stage B realized review requires positive rounds")
    for campaign in campaigns:
        label_name = str(campaign["label_name"])
        labels = label_table(Path(str(campaign["label_table_path"])), label_name=label_name)
        label_by_id = dict(zip(labels["id"], labels[label_name], strict=True))
        pool_baseline = float(labels[label_name].mean())
        workdir = campaign_workdir(Path(str(campaign["config_path"])))
        selected_count_expected = int(campaign.get("selection_k") or 0)
        if selected_count_expected <= 0:
            raise ValueError(f"campaign missing positive selection_k: {campaign.get('campaign_key')}")
        null_metadata = null_metadata_from_label_table(labels)
        seed_summary = seed_label_summary(
            Path(str(campaign["initial_label_input_path"])),
            label_name=label_name,
            pool_baseline=pool_baseline,
        )
        for round_index in range(rounds):
            selection = selection_table(workdir, round_index=round_index)
            selected_ids = [str(value) for value in selection["id"].tolist()]
            reject_duplicate_ids(selected_ids, path=workdir, round_index=round_index)
            missing = sorted(set(selected_ids) - set(label_by_id))
            if missing:
                raise ValueError(
                    "Stage B realized review selected id(s) missing from label table: "
                    f"campaign={campaign.get('campaign_key')}, round={round_index}, sample={missing[:5]}"
                )
            selected_values = np.array([float(label_by_id[candidate_id]) for candidate_id in selected_ids], dtype=float)
            selected_mean = float(np.mean(selected_values))
            lift_ratio = selected_mean / pool_baseline if pool_baseline > 0 else np.nan
            rows.append(
                {
                    "campaign_key": str(campaign["campaign_key"]),
                    "label_name": label_name,
                    "label_family_id": str(campaign["label_family_id"]),
                    "oracle_role": str(campaign["oracle_role"]),
                    "split_id": str(campaign["split_id"]),
                    "seed": int(campaign["seed"]),
                    "initial_seed_policy": str(campaign.get("initial_seed_policy") or ""),
                    "round": int(round_index),
                    "selected_count": int(len(selected_ids)),
                    "selection_k": selected_count_expected,
                    "selection_budget_status": "PASS"
                    if len(selected_ids) == selected_count_expected
                    else "FAIL_SELECTED_COUNT",
                    "selected_true_sum": float(np.sum(selected_values)),
                    "selected_true_mean": selected_mean,
                    "pool_baseline": pool_baseline,
                    **seed_summary,
                    "selected_true_lift_delta": selected_mean - pool_baseline,
                    "selected_true_lift_ratio": lift_ratio,
                    "selected_predicted_score_mean": predicted_score_mean(selection),
                    **null_metadata,
                }
            )
    return pd.DataFrame(rows)


def pair_summary_frame(
    trajectory: pd.DataFrame,
    *,
    campaigns: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Build final positive/null pair summary rows from trajectory rows."""

    rows = []
    campaign_by_key = {str(row["campaign_key"]): row for row in campaigns}
    for pair in pairs:
        positive_key = str(pair["positive_campaign_key"])
        null_key = str(pair["null_campaign_key"])
        if positive_key not in campaign_by_key or null_key not in campaign_by_key:
            raise ValueError(f"Stage B pair references unknown campaign key: {pair}")
        pos = campaign_trajectory(trajectory, positive_key)
        null = campaign_trajectory(trajectory, null_key)
        if len(pos) != len(null):
            raise ValueError(f"positive/null trajectory length mismatch for label {pair.get('label_name')}")
        final_pos = pos.sort_values("round").iloc[-1]
        final_null = null.sort_values("round").iloc[-1]
        positive_mean_round_lift = float(pos["selected_true_lift_ratio"].mean())
        null_mean_round_lift = float(null["selected_true_lift_ratio"].mean())
        positive_trapezoid_auc = normalized_trapezoid_auc(pos)
        null_trapezoid_auc = normalized_trapezoid_auc(null)
        null_claim_status = single_nonempty(null["negative_control_claim_status"].tolist())
        if null_claim_status and null_claim_status != VALID_NEGATIVE_CONTROL:
            peer_status = "null_is_confound_control_only"
        elif final_pos["selected_true_lift_ratio"] > final_null["selected_true_lift_ratio"]:
            peer_status = "positive_exceeds_null"
        else:
            peer_status = "not_separated_from_null"
        rows.append(
            {
                "label_name": str(pair["label_name"]),
                "label_family_id": str(final_pos["label_family_id"]),
                "split_id": str(pair["split_id"]),
                "seed": int(pair["seed"]),
                "positive_campaign_key": positive_key,
                "null_campaign_key": null_key,
                "positive_final_selected_count": int(final_pos["selected_count"]),
                "null_final_selected_count": int(final_null["selected_count"]),
                "positive_final_selected_true_sum": float(final_pos["selected_true_sum"]),
                "null_final_selected_true_sum": float(final_null["selected_true_sum"]),
                "positive_final_selected_true_mean": float(final_pos["selected_true_mean"]),
                "null_final_selected_true_mean": float(final_null["selected_true_mean"]),
                "positive_pool_baseline": float(final_pos["pool_baseline"]),
                "null_pool_baseline": float(final_null["pool_baseline"]),
                "positive_final_lift_ratio": float(final_pos["selected_true_lift_ratio"]),
                "null_final_lift_ratio": float(final_null["selected_true_lift_ratio"]),
                "final_positive_minus_null_lift_ratio": float(
                    final_pos["selected_true_lift_ratio"] - final_null["selected_true_lift_ratio"]
                ),
                "positive_mean_round_lift_ratio": positive_mean_round_lift,
                "null_mean_round_lift_ratio": null_mean_round_lift,
                "mean_round_positive_minus_null_lift_ratio": positive_mean_round_lift - null_mean_round_lift,
                "positive_trapezoid_auc_lift_ratio": positive_trapezoid_auc,
                "null_trapezoid_auc_lift_ratio": null_trapezoid_auc,
                "trapezoid_auc_positive_minus_null_lift_ratio": positive_trapezoid_auc - null_trapezoid_auc,
                "null_control_role": single_nonempty(null["null_control_role"].tolist()),
                "negative_control_claim_status": null_claim_status,
                "peer_review_claim_status": peer_status,
            }
        )
    return pd.DataFrame(rows)


def campaign_trajectory(trajectory: pd.DataFrame, campaign_key: str) -> pd.DataFrame:
    """Return trajectory rows for one campaign key."""

    out = trajectory.loc[trajectory["campaign_key"] == campaign_key].copy()
    if out.empty:
        raise ValueError(f"missing trajectory rows for campaign {campaign_key}")
    return out.sort_values("round")


def normalized_trapezoid_auc(frame: pd.DataFrame) -> float:
    """Return round-normalized trapezoid AUC so the value remains lift-scaled."""

    ordered = frame.sort_values("round")
    rounds = pd.to_numeric(ordered["round"], errors="raise").to_numpy(dtype=float)
    values = pd.to_numeric(ordered["selected_true_lift_ratio"], errors="raise").to_numpy(dtype=float)
    if len(values) == 0:
        raise ValueError("cannot compute Stage B trajectory AUC with no rounds")
    if len(values) == 1:
        return float(values[0])
    span = float(rounds[-1] - rounds[0])
    if span <= 0:
        raise ValueError("Stage B trajectory rounds must increase to compute normalized AUC")
    widths = np.diff(rounds)
    area = np.sum(widths * (values[:-1] + values[1:]) / 2.0)
    return float(area / span)


def seed_label_summary(path: Path, *, label_name: str, pool_baseline: float) -> dict[str, Any]:
    """Summarize the initial seed-label input used before OPAL acquisition."""

    if not path.exists():
        raise FileNotFoundError(f"Stage B initial seed label input missing: {path}")
    frame = pd.read_parquet(path)
    missing = sorted({"id", label_name} - set(frame.columns))
    if missing:
        raise ValueError(f"Stage B initial seed label input missing column(s): {missing}")
    values = pd.to_numeric(frame[label_name], errors="raise")
    seed_mean = float(values.mean())
    return {
        "seed_label_count": int(len(frame)),
        "seed_true_sum": float(values.sum()),
        "seed_true_mean": seed_mean,
        "seed_true_lift_ratio": seed_mean / pool_baseline if pool_baseline > 0 else np.nan,
        "round_zero_semantics": "first_model_selected_batch_after_seed_labels",
    }


def null_metadata_from_label_table(label_table: pd.DataFrame) -> dict[str, str]:
    """Return review metadata for positive/null label tables."""

    explicit_role = single_nonempty(label_table.get("null_control_role", pd.Series(dtype=str)).tolist())
    explicit_status = single_nonempty(label_table.get("negative_control_claim_status", pd.Series(dtype=str)).tolist())
    if explicit_role and explicit_status:
        return {
            "null_control_role": explicit_role,
            "negative_control_claim_status": explicit_status,
        }
    null_version = single_nonempty(label_table.get("null_version", pd.Series(dtype=str)).tolist())
    if null_version == TFBS_LEARNABILITY_FAMILY_CONTENT_NULL_VERSION:
        return {
            "null_control_role": "matched_label_permutation_negative_control",
            "negative_control_claim_status": VALID_NEGATIVE_CONTROL,
        }
    if null_version == TFBS_LEARNABILITY_SLOT_GEOMETRY_NULL_VERSION:
        return {
            "null_control_role": "count_preserving_slot_confound_control",
            "negative_control_claim_status": "CONFOUND_CONTROL_ONLY",
        }
    return {
        "null_control_role": explicit_role,
        "negative_control_claim_status": explicit_status,
    }


def single_nonempty(values: Sequence[Any]) -> str:
    """Return a single non-empty string value or an empty string when ambiguous."""

    clean = sorted({str(value) for value in values if str(value) not in {"", "nan", "None"}})
    return clean[0] if len(clean) == 1 else ""


def predicted_score_mean(selection: pd.DataFrame) -> float:
    """Return the selected prediction score mean when present."""

    if "pred__score_selected" not in selection.columns:
        return np.nan
    return float(pd.to_numeric(selection["pred__score_selected"], errors="coerce").mean())


def reject_duplicate_ids(ids: Sequence[str], *, path: Path, round_index: int) -> None:
    """Fail fast on malformed selection artifacts with duplicate ids."""

    if len(set(ids)) != len(ids):
        raise ValueError(f"Stage B selection artifact has duplicate id(s): workdir={path}, round={round_index}")
