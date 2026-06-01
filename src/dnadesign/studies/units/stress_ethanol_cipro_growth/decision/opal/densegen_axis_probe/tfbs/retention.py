"""Retention preflight estimates for DenseGen TFBS learnability campaigns."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from ..core.constants import DEFAULT_SEED, DEFAULT_SUITE_SEEDS, SPLITS
from .schema import (
    TFBS_LEARNABILITY_MINIMUM_TARGET_SET,
    TFBS_LEARNABILITY_SCHEMA_VERSION,
    TFBS_LEARNABILITY_SENTINEL_TARGET_SET,
)

DEFAULT_RETENTION_MAX_ESTIMATED_BYTES = 50_000_000_000
DEFAULT_TFBS_STAGE_ROUNDS = 24
DEFAULT_TFBS_STAGE_SELECTION_K = 6
DEFAULT_TFBS_STAGE_INITIAL_LABELS = DEFAULT_TFBS_STAGE_SELECTION_K
ORACLE_ROLE_COUNT = 2
RETAINED_FULL_PREDICTION_SNAPSHOTS = ("latest", "final")


@dataclass(frozen=True)
class TfbsRetentionPolicy:
    """Artifact retention policy used by the TFBS learnability preflight."""

    mode: str = "production_review"
    rounds: int = DEFAULT_TFBS_STAGE_ROUNDS
    selection_k: int = DEFAULT_TFBS_STAGE_SELECTION_K
    keep_full_prediction_snapshots: tuple[str, ...] = RETAINED_FULL_PREDICTION_SNAPSHOTS
    keep_selected_row_history: bool = True
    write_all_row_plot_csvs: bool = False
    max_estimated_bytes: int = DEFAULT_RETENTION_MAX_ESTIMATED_BYTES
    fail_if_estimate_exceeds: bool = True

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["keep_full_prediction_snapshots"] = list(self.keep_full_prediction_snapshots)
        return payload


def retention_policy_hash(policy: TfbsRetentionPolicy) -> str:
    """Return the stable content hash used by pairing and Stage A manifests."""

    return _payload_hash(policy.to_dict())


def estimate_tfbs_learnability_retention(
    *,
    candidate_row_count: int,
    policy: TfbsRetentionPolicy | None = None,
) -> dict[str, Any]:
    """Estimate artifact sizes for the sentinel and full TFBS learnability matrices."""

    cfg = policy or TfbsRetentionPolicy()
    _validate_policy(cfg)
    row_count = int(candidate_row_count)
    if row_count <= 0:
        raise ValueError("candidate_row_count must be positive")

    estimates = {
        "sentinel_initial": _estimate_matrix(
            name="sentinel_initial",
            label_count=len(TFBS_LEARNABILITY_SENTINEL_TARGET_SET),
            split_count=1,
            seed_count=1,
            candidate_row_count=row_count,
            policy=cfg,
        ),
        "full_matrix": _estimate_matrix(
            name="full_matrix",
            label_count=len(TFBS_LEARNABILITY_MINIMUM_TARGET_SET),
            split_count=len(SPLITS),
            seed_count=len(DEFAULT_SUITE_SEEDS),
            candidate_row_count=row_count,
            policy=cfg,
        ),
    }
    max_expected_total_bytes = max(row["expected_total_bytes"] for row in estimates.values())
    status = "PASS" if max_expected_total_bytes <= cfg.max_estimated_bytes else "FAIL_BUDGET_EXCEEDED"
    payload = {
        "schema_version": f"{TFBS_LEARNABILITY_SCHEMA_VERSION}.retention_estimate",
        "status": status,
        "retention_policy_hash": retention_policy_hash(cfg),
        "retention_policy": cfg.to_dict(),
        "candidate_row_count": row_count,
        "max_estimated_bytes": int(cfg.max_estimated_bytes),
        "fail_if_estimate_exceeds": bool(cfg.fail_if_estimate_exceeds),
        "estimates": estimates,
        "max_expected_total_bytes": int(max_expected_total_bytes),
        "budget_margin_bytes": int(cfg.max_estimated_bytes - max_expected_total_bytes),
        "sentinel_labels": list(TFBS_LEARNABILITY_SENTINEL_TARGET_SET),
        "full_matrix_labels": list(TFBS_LEARNABILITY_MINIMUM_TARGET_SET),
        "initial_sentinel_seed": DEFAULT_SEED,
        "full_matrix_seeds": list(DEFAULT_SUITE_SEEDS),
        "full_matrix_splits": list(SPLITS),
    }
    if status != "PASS" and cfg.fail_if_estimate_exceeds:
        raise ValueError(
            "retention estimate exceeds configured budget: "
            f"expected={max_expected_total_bytes} max={cfg.max_estimated_bytes}"
        )
    return payload


def _estimate_matrix(
    *,
    name: str,
    label_count: int,
    split_count: int,
    seed_count: int,
    candidate_row_count: int,
    policy: TfbsRetentionPolicy,
) -> dict[str, Any]:
    planned_campaign_count = int(label_count * ORACLE_ROLE_COUNT * split_count * seed_count)
    round_count = int(policy.rounds)
    label_dimension = 1
    retained_snapshot_count = len(tuple(dict.fromkeys(policy.keep_full_prediction_snapshots)))
    full_prediction_rows = planned_campaign_count * candidate_row_count * retained_snapshot_count
    full_prediction_y_hat_cells = full_prediction_rows * label_dimension
    all_round_full_prediction_rows = planned_campaign_count * round_count * candidate_row_count
    selected_row_ledger_rows = planned_campaign_count * round_count * int(policy.selection_k)
    if policy.write_all_row_plot_csvs:
        plot_derived_table_rows = all_round_full_prediction_rows
    else:
        plot_derived_table_rows = selected_row_ledger_rows

    expected_prediction_ledger_bytes = full_prediction_rows * 96 + selected_row_ledger_rows * 144
    expected_plot_data_bytes = plot_derived_table_rows * 128
    expected_model_artifact_bytes = planned_campaign_count * round_count * 50_000
    expected_manifest_bytes = planned_campaign_count * 20_000 + label_count * 8_000
    expected_total_bytes = (
        expected_prediction_ledger_bytes
        + expected_plot_data_bytes
        + expected_model_artifact_bytes
        + expected_manifest_bytes
    )
    return {
        "name": name,
        "planned_campaign_count": planned_campaign_count,
        "round_count": round_count,
        "candidate_row_count": int(candidate_row_count),
        "label_dimension": label_dimension,
        "label_count": int(label_count),
        "oracle_role_count": ORACLE_ROLE_COUNT,
        "split_count": int(split_count),
        "seed_count": int(seed_count),
        "retained_full_prediction_snapshot_count": retained_snapshot_count,
        "full_prediction_rows": int(full_prediction_rows),
        "all_round_full_prediction_rows_if_audit_full": int(all_round_full_prediction_rows),
        "full_prediction_y_hat_cells": int(full_prediction_y_hat_cells),
        "selected_row_ledger_rows": int(selected_row_ledger_rows),
        "plot_derived_table_rows": int(plot_derived_table_rows),
        "expected_prediction_ledger_bytes": int(expected_prediction_ledger_bytes),
        "expected_plot_data_bytes": int(expected_plot_data_bytes),
        "expected_model_artifact_bytes": int(expected_model_artifact_bytes),
        "expected_manifest_bytes": int(expected_manifest_bytes),
        "expected_total_bytes": int(expected_total_bytes),
    }


def _validate_policy(policy: TfbsRetentionPolicy) -> None:
    if policy.mode != "production_review":
        raise ValueError(f"unsupported TFBS retention mode for Stage A: {policy.mode!r}")
    if policy.rounds <= 0:
        raise ValueError("retention rounds must be positive")
    if policy.selection_k <= 0:
        raise ValueError("retention selection_k must be positive")
    if policy.max_estimated_bytes <= 0:
        raise ValueError("max_estimated_bytes must be positive")
    unknown = sorted(set(policy.keep_full_prediction_snapshots) - {"latest", "final"})
    if unknown:
        raise ValueError(f"unsupported full-prediction snapshot marker(s): {unknown}")


def _payload_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"
