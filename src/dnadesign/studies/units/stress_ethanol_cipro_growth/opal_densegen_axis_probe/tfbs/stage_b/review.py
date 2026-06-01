"""Realized-label review artifacts for DenseGen TFBS Stage B campaigns."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..schema import (
    TFBS_LEARNABILITY_FAMILY_CONTENT_NULL_VERSION,
    TFBS_LEARNABILITY_SLOT_GEOMETRY_NULL_VERSION,
)
from ..stage_a.manifests import file_sha256
from .claims import (
    build_tfbs_stage_b_claim_assessment,
    summarize_tfbs_stage_b_claim_assessment,
)
from .notebook_visuals import (
    maybe_register_tfbs_stage_b_realized_review_visuals,
    maybe_register_tfbs_stage_b_slot_diagnostic_visuals,
)
from .review_plots import materialize_tfbs_stage_b_realized_review_plots
from .slot_diagnostics import SLOT_LABEL_SPECS, build_tfbs_stage_b_slot_diagnostics

REALIZED_REVIEW_SCHEMA_VERSION = "stress_ethanol_cipro_growth.tfbs_stage_b_realized_review.v1"
VALID_NEGATIVE_CONTROL = "VALID_AS_NEGATIVE_CONTROL"


@dataclass(frozen=True)
class TfbsStageBRealizedReviewResult:
    """Paths for a materialized realized-label Stage B review."""

    status: str
    review_dir: Path
    trajectory_csv_path: Path
    pair_summary_csv_path: Path
    claim_assessment_csv_path: Path
    plot_manifest_json_path: Path
    slot_diagnostics_summary_json_path: Path | None
    slot_diagnostics_plot_manifest_json_path: Path | None
    notebook_visual_registration: Mapping[str, Any]
    summary_json_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "review_dir": str(self.review_dir),
            "trajectory_csv_path": str(self.trajectory_csv_path),
            "pair_summary_csv_path": str(self.pair_summary_csv_path),
            "claim_assessment_csv_path": str(self.claim_assessment_csv_path),
            "plot_manifest_json_path": str(self.plot_manifest_json_path),
            "slot_diagnostics_summary_json_path": (
                str(self.slot_diagnostics_summary_json_path)
                if self.slot_diagnostics_summary_json_path is not None
                else None
            ),
            "slot_diagnostics_plot_manifest_json_path": (
                str(self.slot_diagnostics_plot_manifest_json_path)
                if self.slot_diagnostics_plot_manifest_json_path is not None
                else None
            ),
            "notebook_visual_registration": dict(self.notebook_visual_registration),
            "summary_json_path": str(self.summary_json_path),
        }


def build_tfbs_stage_b_realized_label_review(
    config_manifest_path: str | Path,
    *,
    out_dir: str | Path | None = None,
    collection_visual_index_path: str | Path | None = None,
) -> TfbsStageBRealizedReviewResult:
    """Write true-label trajectory and paired positive/null review artifacts."""

    manifest_path = Path(config_manifest_path)
    manifest = _read_json(manifest_path)
    campaigns = _campaign_rows(manifest)
    review_dir = Path(out_dir) if out_dir is not None else manifest_path.parent.parent / "review" / "realized_labels"
    review_dir.mkdir(parents=True, exist_ok=True)

    trajectory = _trajectory_frame(campaigns, rounds=int(manifest["rounds"]))
    pair_summary = _pair_summary_frame(trajectory, campaigns=campaigns, pairs=_pair_rows(manifest))
    claim_assessment = build_tfbs_stage_b_claim_assessment(pair_summary)
    trajectory_path = review_dir / "tfbs_stage_b_realized_label_trajectory.csv"
    pair_summary_path = review_dir / "tfbs_stage_b_positive_null_pair_summary.csv"
    claim_assessment_path = review_dir / "tfbs_stage_b_claim_assessment.csv"
    summary_path = review_dir / "tfbs_stage_b_realized_label_review.json"
    trajectory.to_csv(trajectory_path, index=False)
    pair_summary.to_csv(pair_summary_path, index=False)
    claim_assessment.to_csv(claim_assessment_path, index=False)
    plot_manifest_path = materialize_tfbs_stage_b_realized_review_plots(
        trajectory_csv_path=trajectory_path,
        pair_summary_csv_path=pair_summary_path,
        out_dir=review_dir / "plots",
    )
    slot_diagnostics = (
        build_tfbs_stage_b_slot_diagnostics(
            manifest_path,
            out_dir=review_dir,
        )
        if _has_slot_pairs(manifest)
        else None
    )
    realized_visual_registration = maybe_register_tfbs_stage_b_realized_review_visuals(
        config_manifest_path=manifest_path,
        plot_manifest_json_path=plot_manifest_path,
        trajectory_csv_path=trajectory_path,
        pair_summary_csv_path=pair_summary_path,
        collection_visual_index_path=collection_visual_index_path,
    )
    slot_visual_registration = (
        maybe_register_tfbs_stage_b_slot_diagnostic_visuals(
            config_manifest_path=manifest_path,
            plot_manifest_json_path=slot_diagnostics.plot_manifest_json_path,
            trajectory_csv_path=slot_diagnostics.trajectory_csv_path,
            pair_summary_csv_path=slot_diagnostics.pair_summary_csv_path,
            count_distribution_csv_path=slot_diagnostics.count_distribution_csv_path,
            collection_visual_index_path=collection_visual_index_path,
        )
        if slot_diagnostics is not None
        else {
            "status": "SKIPPED_NO_SLOT_LABELS",
            "collection_visual_index_path": None,
            "registered_visual_count": 0,
        }
    )
    notebook_visual_registration = {
        "realized_label_review": dict(realized_visual_registration),
        "slot_count_diagnostics": dict(slot_visual_registration),
    }
    summary = _summary_payload(
        manifest_path=manifest_path,
        manifest=manifest,
        trajectory_path=trajectory_path,
        pair_summary_path=pair_summary_path,
        claim_assessment_path=claim_assessment_path,
        plot_manifest_path=plot_manifest_path,
        slot_diagnostics_summary_path=slot_diagnostics.summary_json_path if slot_diagnostics is not None else None,
        slot_diagnostics_plot_manifest_path=slot_diagnostics.plot_manifest_json_path
        if slot_diagnostics is not None
        else None,
        notebook_visual_registration=notebook_visual_registration,
        trajectory=trajectory,
        pair_summary=pair_summary,
        claim_assessment=claim_assessment,
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return TfbsStageBRealizedReviewResult(
        status=str(summary["status"]),
        review_dir=review_dir,
        trajectory_csv_path=trajectory_path,
        pair_summary_csv_path=pair_summary_path,
        claim_assessment_csv_path=claim_assessment_path,
        plot_manifest_json_path=plot_manifest_path,
        slot_diagnostics_summary_json_path=slot_diagnostics.summary_json_path if slot_diagnostics is not None else None,
        slot_diagnostics_plot_manifest_json_path=slot_diagnostics.plot_manifest_json_path
        if slot_diagnostics is not None
        else None,
        notebook_visual_registration=notebook_visual_registration,
        summary_json_path=summary_path,
    )


def _trajectory_frame(campaigns: Sequence[Mapping[str, Any]], *, rounds: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if rounds <= 0:
        raise ValueError("Stage B realized review requires positive rounds")
    for campaign in campaigns:
        label_name = str(campaign["label_name"])
        label_table = _label_table(Path(str(campaign["label_table_path"])), label_name=label_name)
        label_by_id = dict(zip(label_table["id"], label_table[label_name], strict=True))
        pool_baseline = float(label_table[label_name].mean())
        workdir = _campaign_workdir(Path(str(campaign["config_path"])))
        selected_count_expected = int(campaign.get("selection_k") or 0)
        if selected_count_expected <= 0:
            raise ValueError(f"campaign missing positive selection_k: {campaign.get('campaign_key')}")
        null_metadata = _null_metadata(label_table)
        seed_summary = _seed_label_summary(
            Path(str(campaign["initial_label_input_path"])),
            label_name=label_name,
            pool_baseline=pool_baseline,
        )
        for round_index in range(rounds):
            selection = _selection_table(workdir, round_index=round_index)
            selected_ids = [str(value) for value in selection["id"].tolist()]
            _reject_duplicate_ids(selected_ids, path=workdir, round_index=round_index)
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
                    "selected_predicted_score_mean": _predicted_score_mean(selection),
                    **null_metadata,
                }
            )
    return pd.DataFrame(rows)


def _pair_summary_frame(
    trajectory: pd.DataFrame,
    *,
    campaigns: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    rows = []
    campaign_by_key = {str(row["campaign_key"]): row for row in campaigns}
    for pair in pairs:
        positive_key = str(pair["positive_campaign_key"])
        null_key = str(pair["null_campaign_key"])
        if positive_key not in campaign_by_key or null_key not in campaign_by_key:
            raise ValueError(f"Stage B pair references unknown campaign key: {pair}")
        pos = _campaign_trajectory(trajectory, positive_key)
        null = _campaign_trajectory(trajectory, null_key)
        if len(pos) != len(null):
            raise ValueError(f"positive/null trajectory length mismatch for label {pair.get('label_name')}")
        final_pos = pos.sort_values("round").iloc[-1]
        final_null = null.sort_values("round").iloc[-1]
        positive_mean_round_lift = float(pos["selected_true_lift_ratio"].mean())
        null_mean_round_lift = float(null["selected_true_lift_ratio"].mean())
        positive_trapezoid_auc = _normalized_trapezoid_auc(pos)
        null_trapezoid_auc = _normalized_trapezoid_auc(null)
        null_claim_status = _single_nonempty(null["negative_control_claim_status"].tolist())
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
                "null_control_role": _single_nonempty(null["null_control_role"].tolist()),
                "negative_control_claim_status": null_claim_status,
                "peer_review_claim_status": peer_status,
            }
        )
    return pd.DataFrame(rows)


def _summary_payload(
    *,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    trajectory_path: Path,
    pair_summary_path: Path,
    claim_assessment_path: Path,
    plot_manifest_path: Path,
    slot_diagnostics_summary_path: Path | None,
    slot_diagnostics_plot_manifest_path: Path | None,
    notebook_visual_registration: Mapping[str, Any],
    trajectory: pd.DataFrame,
    pair_summary: pd.DataFrame,
    claim_assessment: pd.DataFrame,
) -> dict[str, Any]:
    budget_failures = int((trajectory["selection_budget_status"] != "PASS").sum()) if not trajectory.empty else 0
    confounded_pairs = (
        int((pair_summary["peer_review_claim_status"] == "null_is_confound_control_only").sum())
        if not pair_summary.empty
        else 0
    )
    return {
        "schema_version": REALIZED_REVIEW_SCHEMA_VERSION,
        "status": "PASS" if budget_failures == 0 else "FAIL_SELECTION_BUDGET",
        "source_config_manifest_path": str(manifest_path),
        "source_config_manifest_hash": file_sha256(manifest_path),
        "campaign_count": int(manifest["campaign_count"]),
        "pair_count": int(len(pair_summary)),
        "rounds": int(manifest["rounds"]),
        "trajectory_csv_path": str(trajectory_path),
        "trajectory_csv_hash": file_sha256(trajectory_path),
        "pair_summary_csv_path": str(pair_summary_path),
        "pair_summary_csv_hash": file_sha256(pair_summary_path),
        "claim_assessment_csv_path": str(claim_assessment_path),
        "claim_assessment_csv_hash": file_sha256(claim_assessment_path),
        "claim_readiness": summarize_tfbs_stage_b_claim_assessment(claim_assessment),
        "plot_manifest_json_path": str(plot_manifest_path),
        "plot_manifest_json_hash": file_sha256(plot_manifest_path),
        "slot_diagnostics_summary_json_path": (
            str(slot_diagnostics_summary_path) if slot_diagnostics_summary_path is not None else None
        ),
        "slot_diagnostics_summary_json_hash": (
            file_sha256(slot_diagnostics_summary_path) if slot_diagnostics_summary_path is not None else None
        ),
        "slot_diagnostics_plot_manifest_json_path": (
            str(slot_diagnostics_plot_manifest_path) if slot_diagnostics_plot_manifest_path is not None else None
        ),
        "slot_diagnostics_plot_manifest_json_hash": (
            file_sha256(slot_diagnostics_plot_manifest_path)
            if slot_diagnostics_plot_manifest_path is not None
            else None
        ),
        "notebook_visual_registration": dict(notebook_visual_registration),
        "budget_failure_count": budget_failures,
        "confounded_null_pair_count": confounded_pairs,
        "interpretation_boundary": (
            "Realized selected-label lift is the primary ML learnability endpoint. "
            "Predicted selected score is an acquisition diagnostic and must not be used alone as evidence "
            "that a positive oracle is learned better than its null/control."
        ),
    }


def _campaign_trajectory(trajectory: pd.DataFrame, campaign_key: str) -> pd.DataFrame:
    out = trajectory.loc[trajectory["campaign_key"] == campaign_key].copy()
    if out.empty:
        raise ValueError(f"missing trajectory rows for campaign {campaign_key}")
    return out.sort_values("round")


def _normalized_trapezoid_auc(frame: pd.DataFrame) -> float:
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


def _campaign_rows(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if manifest.get("status") != "PASS":
        raise ValueError("Stage B realized review requires config manifest status PASS")
    rows = manifest.get("campaigns")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Stage B realized review requires non-empty campaigns")
    return [row for row in rows if isinstance(row, Mapping)]


def _pair_rows(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = manifest.get("pairs")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Stage B realized review requires non-empty positive/null pairs")
    return [row for row in rows if isinstance(row, Mapping)]


def _has_slot_pairs(manifest: Mapping[str, Any]) -> bool:
    return any(str(row.get("label_name")) in SLOT_LABEL_SPECS for row in _pair_rows(manifest))


def _label_table(path: Path, *, label_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Stage B label table not found: {path}")
    frame = pd.read_parquet(path)
    missing = sorted({"id", label_name} - set(frame.columns))
    if missing:
        raise ValueError(f"Stage B label table missing column(s): {missing}")
    out = frame.copy()
    out["id"] = out["id"].astype(str)
    if out["id"].duplicated().any():
        duplicates = out.loc[out["id"].duplicated(), "id"].head(5).tolist()
        raise ValueError(f"Stage B label table contains duplicate id(s): {duplicates}")
    out[label_name] = pd.to_numeric(out[label_name], errors="raise")
    return out


def _selection_table(workdir: Path, *, round_index: int) -> pd.DataFrame:
    path = workdir / "outputs" / "rounds" / f"round_{int(round_index)}" / "selection" / "selection_top_k.csv"
    if not path.exists():
        raise FileNotFoundError(f"Stage B selection artifact missing: {path}")
    frame = pd.read_csv(path)
    if "id" not in frame.columns:
        raise ValueError(f"Stage B selection artifact missing id column: {path}")
    frame["id"] = frame["id"].astype(str)
    return frame


def _seed_label_summary(path: Path, *, label_name: str, pool_baseline: float) -> dict[str, Any]:
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


def _campaign_workdir(config_path: Path) -> Path:
    if config_path.name != "campaign.yaml" or config_path.parent.name != "configs":
        raise ValueError(f"Stage B config path does not follow campaign/configs/campaign.yaml layout: {config_path}")
    return config_path.parent.parent


def _null_metadata(label_table: pd.DataFrame) -> dict[str, str]:
    explicit_role = _single_nonempty(label_table.get("null_control_role", pd.Series(dtype=str)).tolist())
    explicit_status = _single_nonempty(label_table.get("negative_control_claim_status", pd.Series(dtype=str)).tolist())
    if explicit_role and explicit_status:
        return {
            "null_control_role": explicit_role,
            "negative_control_claim_status": explicit_status,
        }
    null_version = _single_nonempty(label_table.get("null_version", pd.Series(dtype=str)).tolist())
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


def _single_nonempty(values: Sequence[Any]) -> str:
    clean = sorted({str(value) for value in values if str(value) not in {"", "nan", "None"}})
    return clean[0] if len(clean) == 1 else ""


def _predicted_score_mean(selection: pd.DataFrame) -> float:
    if "pred__score_selected" not in selection.columns:
        return np.nan
    return float(pd.to_numeric(selection["pred__score_selected"], errors="coerce").mean())


def _reject_duplicate_ids(ids: Sequence[str], *, path: Path, round_index: int) -> None:
    if len(set(ids)) != len(ids):
        raise ValueError(f"Stage B selection artifact has duplicate id(s): workdir={path}, round={round_index}")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Stage B config manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Stage B config manifest must be a JSON object: {path}")
    return payload
