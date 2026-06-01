"""Slot-count confound diagnostics for DenseGen TFBS Stage B campaigns."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..schema import TFBS_LEARNABILITY_SCHEMA_VERSION
from ..stage_a.manifests import file_sha256
from .slot_plots import materialize_tfbs_stage_b_slot_diagnostic_plots

SLOT_DIAGNOSTIC_SCHEMA_VERSION = f"{TFBS_LEARNABILITY_SCHEMA_VERSION}.stage_b_slot_diagnostics"
MAX_TFBS_SLOT_COUNT = 3
POSITION_SIGNAL_AFTER_COUNT_RESTRICTION = "position_signal_after_count_restriction"
NOT_SEPARATED_AFTER_COUNT_RESTRICTION = "not_separated_after_count_restriction"
INSUFFICIENT_NONDETERMINISTIC_SELECTION = "insufficient_nondeterministic_selection"


@dataclass(frozen=True)
class SlotLabelSpec:
    """Count column and deterministic strata for a slot-family target label."""

    label_name: str
    target_family_count_column: str
    max_target_family_count: int = MAX_TFBS_SLOT_COUNT

    @property
    def deterministic_counts(self) -> tuple[int, int]:
        return (0, int(self.max_target_family_count))


@dataclass(frozen=True)
class TfbsStageBSlotDiagnosticResult:
    """Paths for a materialized Stage B slot-count diagnostic bundle."""

    status: str
    review_dir: Path
    trajectory_csv_path: Path
    count_distribution_csv_path: Path
    pair_summary_csv_path: Path
    plot_manifest_json_path: Path
    summary_json_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "review_dir": str(self.review_dir),
            "trajectory_csv_path": str(self.trajectory_csv_path),
            "count_distribution_csv_path": str(self.count_distribution_csv_path),
            "pair_summary_csv_path": str(self.pair_summary_csv_path),
            "plot_manifest_json_path": str(self.plot_manifest_json_path),
            "summary_json_path": str(self.summary_json_path),
        }


SLOT_LABEL_SPECS: dict[str, SlotLabelSpec] = {
    "lexA_in_slot0": SlotLabelSpec("lexA_in_slot0", "lexA_count"),
    "lexA_in_slot1": SlotLabelSpec("lexA_in_slot1", "lexA_count"),
    "lexA_in_slot2": SlotLabelSpec("lexA_in_slot2", "lexA_count"),
    "cpxR_or_baeR_in_slot0": SlotLabelSpec("cpxR_or_baeR_in_slot0", "cpxR_or_baeR_count"),
    "cpxR_or_baeR_in_slot1": SlotLabelSpec("cpxR_or_baeR_in_slot1", "cpxR_or_baeR_count"),
    "cpxR_or_baeR_in_slot2": SlotLabelSpec("cpxR_or_baeR_in_slot2", "cpxR_or_baeR_count"),
}


def build_tfbs_stage_b_slot_diagnostics(
    config_manifest_path: str | Path,
    *,
    out_dir: str | Path | None = None,
) -> TfbsStageBSlotDiagnosticResult:
    """Write count-confound and count-stratified lift diagnostics for slot-label campaigns."""

    manifest_path = Path(config_manifest_path)
    manifest = _read_json(manifest_path)
    campaigns = _campaign_rows(manifest)
    pairs = _slot_pair_rows(manifest)
    review_dir = Path(out_dir) if out_dir is not None else manifest_path.parent.parent / "review" / "realized_labels"
    review_dir.mkdir(parents=True, exist_ok=True)

    trajectory, count_distribution = _slot_trajectory_frames(campaigns, pairs, rounds=int(manifest["rounds"]))
    pair_summary = _slot_pair_summary_frame(trajectory, pairs=pairs)

    trajectory_path = review_dir / "tfbs_stage_b_slot_count_diagnostic_trajectory.csv"
    count_distribution_path = review_dir / "tfbs_stage_b_slot_count_distribution.csv"
    pair_summary_path = review_dir / "tfbs_stage_b_slot_restricted_pair_summary.csv"
    summary_path = review_dir / "tfbs_stage_b_slot_diagnostics.json"
    trajectory.to_csv(trajectory_path, index=False)
    count_distribution.to_csv(count_distribution_path, index=False)
    pair_summary.to_csv(pair_summary_path, index=False)
    plot_manifest_path = materialize_tfbs_stage_b_slot_diagnostic_plots(
        trajectory_csv_path=trajectory_path,
        pair_summary_csv_path=pair_summary_path,
        count_distribution_csv_path=count_distribution_path,
        out_dir=review_dir / "plots",
    )
    summary = _summary_payload(
        manifest_path=manifest_path,
        manifest=manifest,
        trajectory_path=trajectory_path,
        count_distribution_path=count_distribution_path,
        pair_summary_path=pair_summary_path,
        plot_manifest_path=plot_manifest_path,
        trajectory=trajectory,
        pair_summary=pair_summary,
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return TfbsStageBSlotDiagnosticResult(
        status=str(summary["status"]),
        review_dir=review_dir,
        trajectory_csv_path=trajectory_path,
        count_distribution_csv_path=count_distribution_path,
        pair_summary_csv_path=pair_summary_path,
        plot_manifest_json_path=plot_manifest_path,
        summary_json_path=summary_path,
    )


def _slot_trajectory_frames(
    campaigns: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
    *,
    rounds: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if rounds <= 0:
        raise ValueError("Stage B slot diagnostics require positive rounds")
    campaigns_by_key = {str(row["campaign_key"]): row for row in campaigns}
    trajectory_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    for pair in pairs:
        label_name = str(pair["label_name"])
        spec = SLOT_LABEL_SPECS[label_name]
        for role_key in ("positive_campaign_key", "null_campaign_key"):
            campaign_key = str(pair[role_key])
            if campaign_key not in campaigns_by_key:
                raise ValueError(f"Stage B slot pair references unknown campaign key: {campaign_key}")
            campaign = campaigns_by_key[campaign_key]
            label_table = _slot_label_table(Path(str(campaign["label_table_path"])), spec=spec)
            label_by_id = dict(zip(label_table["id"], label_table[label_name], strict=True))
            count_by_id = dict(zip(label_table["id"], label_table[spec.target_family_count_column], strict=True))
            baseline_by_count = _baseline_by_count(label_table, spec=spec)
            pool_rows = _pool_count_distribution(label_table, spec=spec)
            pool_label_baseline = float(label_table[label_name].mean())
            nondeterministic_pool = _nondeterministic_frame(label_table, spec=spec)
            pool_nondeterministic_baseline = (
                float(nondeterministic_pool[label_name].mean()) if not nondeterministic_pool.empty else np.nan
            )
            pool_target_count_mean = float(label_table[spec.target_family_count_column].mean())
            pool_deterministic_fraction = float(_deterministic_mask(label_table, spec=spec).mean())
            workdir = _campaign_workdir(Path(str(campaign["config_path"])))
            for round_index in range(rounds):
                selection = _selection_table(workdir, round_index=round_index)
                selected_ids = [str(value) for value in selection["id"].tolist()]
                _reject_duplicate_ids(selected_ids, path=workdir, round_index=round_index)
                missing = sorted(set(selected_ids) - set(label_by_id))
                if missing:
                    raise ValueError(
                        "Stage B slot diagnostics selected id(s) missing from label table: "
                        f"campaign={campaign_key}, round={round_index}, sample={missing[:5]}"
                    )
                selected = pd.DataFrame(
                    {
                        "id": selected_ids,
                        label_name: [float(label_by_id[candidate_id]) for candidate_id in selected_ids],
                        spec.target_family_count_column: [
                            int(count_by_id[candidate_id]) for candidate_id in selected_ids
                        ],
                    }
                )
                selected_mask = _deterministic_mask(selected, spec=spec)
                selected_nondeterministic = selected.loc[~selected_mask].copy()
                count_stratified_expected = _selected_count_stratified_expected(
                    selected_nondeterministic,
                    spec=spec,
                    baseline_by_count=baseline_by_count,
                )
                selected_nd_mean = (
                    float(selected_nondeterministic[label_name].mean())
                    if not selected_nondeterministic.empty
                    else np.nan
                )
                selected_true_mean = float(selected[label_name].mean())
                trajectory_rows.append(
                    {
                        "campaign_key": campaign_key,
                        "label_name": label_name,
                        "label_family_id": str(campaign["label_family_id"]),
                        "oracle_role": str(campaign["oracle_role"]),
                        "split_id": str(campaign["split_id"]),
                        "seed": int(campaign["seed"]),
                        "round": int(round_index),
                        "selected_count": int(len(selected)),
                        "selection_k": int(campaign.get("selection_k") or 0),
                        "target_family_count_column": spec.target_family_count_column,
                        "max_target_family_count": int(spec.max_target_family_count),
                        "deterministic_count_values": json.dumps(list(spec.deterministic_counts)),
                        "pool_label_baseline": pool_label_baseline,
                        "pool_nondeterministic_baseline": pool_nondeterministic_baseline,
                        "pool_target_count_mean": pool_target_count_mean,
                        "pool_deterministic_fraction": pool_deterministic_fraction,
                        "selected_target_count_mean": float(selected[spec.target_family_count_column].mean()),
                        "selected_deterministic_fraction": float(selected_mask.mean()),
                        "selected_true_mean": selected_true_mean,
                        "selected_true_lift_ratio": _safe_ratio(selected_true_mean, pool_label_baseline),
                        "selected_nondeterministic_count": int(len(selected_nondeterministic)),
                        "selected_nondeterministic_true_mean": selected_nd_mean,
                        "nondeterministic_lift_ratio": _safe_ratio(
                            selected_nd_mean,
                            pool_nondeterministic_baseline,
                        ),
                        "count_stratified_expected_baseline": count_stratified_expected,
                        "count_stratified_lift_ratio": _safe_ratio(
                            selected_nd_mean,
                            count_stratified_expected,
                        ),
                        "count_stratified_lift_delta": selected_nd_mean - count_stratified_expected
                        if _is_finite(selected_nd_mean) and _is_finite(count_stratified_expected)
                        else np.nan,
                    }
                )
                distribution_rows.extend(
                    _count_distribution_rows(
                        campaign=campaign,
                        spec=spec,
                        label_name=label_name,
                        round_index=round_index,
                        selected=selected,
                        pool_rows=pool_rows,
                    )
                )
    return pd.DataFrame(trajectory_rows), pd.DataFrame(distribution_rows)


def _slot_pair_summary_frame(
    trajectory: pd.DataFrame,
    *,
    pairs: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        label_name = str(pair["label_name"])
        positive_key = str(pair["positive_campaign_key"])
        null_key = str(pair["null_campaign_key"])
        pos = _campaign_trajectory(trajectory, positive_key)
        null = _campaign_trajectory(trajectory, null_key)
        paired = pos.merge(
            null,
            on="round",
            how="inner",
            suffixes=("_positive", "_null"),
            validate="one_to_one",
        )
        final = paired.sort_values("round").iloc[-1]
        final_delta = float(final["count_stratified_lift_ratio_positive"] - final["count_stratified_lift_ratio_null"])
        auc_positive, auc_null, auc_delta, valid_round_count = _paired_auc_delta(
            paired,
            positive_column="count_stratified_lift_ratio_positive",
            null_column="count_stratified_lift_ratio_null",
        )
        status = _slot_diagnostic_status(
            final_delta=final_delta,
            auc_delta=auc_delta,
            valid_round_count=valid_round_count,
        )
        rows.append(
            {
                "label_name": label_name,
                "split_id": str(pair["split_id"]),
                "seed": int(pair["seed"]),
                "positive_campaign_key": positive_key,
                "null_campaign_key": null_key,
                "positive_final_count_stratified_lift_ratio": float(final["count_stratified_lift_ratio_positive"]),
                "null_final_count_stratified_lift_ratio": float(final["count_stratified_lift_ratio_null"]),
                "final_positive_minus_null_count_stratified_lift_ratio": final_delta,
                "positive_auc_count_stratified_lift_ratio": auc_positive,
                "null_auc_count_stratified_lift_ratio": auc_null,
                "auc_positive_minus_null_count_stratified_lift_ratio": auc_delta,
                "paired_valid_auc_round_count": int(valid_round_count),
                "positive_final_selected_target_count_mean": float(final["selected_target_count_mean_positive"]),
                "null_final_selected_target_count_mean": float(final["selected_target_count_mean_null"]),
                "pool_target_count_mean": float(final["pool_target_count_mean_positive"]),
                "positive_final_selected_deterministic_fraction": float(
                    final["selected_deterministic_fraction_positive"]
                ),
                "null_final_selected_deterministic_fraction": float(final["selected_deterministic_fraction_null"]),
                "slot_diagnostic_status": status,
                "interpretation_boundary": (
                    "Count-stratified lift controls for selected target-family count composition. "
                    "This is a slot-position diagnostic; it does not upgrade a count-preserving slot null into a "
                    "clean negative control."
                ),
            }
        )
    return pd.DataFrame(rows)


def _summary_payload(
    *,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    trajectory_path: Path,
    count_distribution_path: Path,
    pair_summary_path: Path,
    plot_manifest_path: Path,
    trajectory: pd.DataFrame,
    pair_summary: pd.DataFrame,
) -> dict[str, Any]:
    resolved = pair_summary.loc[
        pair_summary["slot_diagnostic_status"] == POSITION_SIGNAL_AFTER_COUNT_RESTRICTION, "label_name"
    ].astype(str)
    unresolved = pair_summary.loc[
        pair_summary["slot_diagnostic_status"] != POSITION_SIGNAL_AFTER_COUNT_RESTRICTION, "label_name"
    ].astype(str)
    return {
        "schema_version": SLOT_DIAGNOSTIC_SCHEMA_VERSION,
        "status": "PASS",
        "source_config_manifest_path": str(manifest_path),
        "source_config_manifest_hash": file_sha256(manifest_path),
        "campaign_count": int(manifest["campaign_count"]),
        "slot_label_count": int(pair_summary["label_name"].nunique()),
        "rounds": int(manifest["rounds"]),
        "trajectory_csv_path": str(trajectory_path),
        "trajectory_csv_hash": file_sha256(trajectory_path),
        "count_distribution_csv_path": str(count_distribution_path),
        "count_distribution_csv_hash": file_sha256(count_distribution_path),
        "pair_summary_csv_path": str(pair_summary_path),
        "pair_summary_csv_hash": file_sha256(pair_summary_path),
        "plot_manifest_json_path": str(plot_manifest_path),
        "plot_manifest_json_hash": file_sha256(plot_manifest_path),
        "resolved_position_signal_labels": resolved.tolist(),
        "unresolved_slot_labels": unresolved.tolist(),
        "slot_diagnostic_status_counts": {
            str(key): int(value)
            for key, value in pair_summary["slot_diagnostic_status"].value_counts().sort_index().to_dict().items()
        },
        "trajectory_row_count": int(len(trajectory)),
        "interpretation_boundary": (
            "The slot null preserves row-level target-family count, so raw slot lift can be count-confounded. "
            "Use the count-stratified diagnostics to decide whether selected rows show position signal beyond count."
        ),
    }


def _slot_pair_rows(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = manifest.get("pairs")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Stage B slot diagnostics require non-empty positive/null pairs")
    slot_rows = [row for row in rows if isinstance(row, Mapping) and str(row.get("label_name")) in SLOT_LABEL_SPECS]
    if not slot_rows:
        raise ValueError("Stage B slot diagnostics require at least one slot-family label pair")
    return slot_rows


def _campaign_rows(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if manifest.get("status") != "PASS":
        raise ValueError("Stage B slot diagnostics require config manifest status PASS")
    rows = manifest.get("campaigns")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Stage B slot diagnostics require non-empty campaigns")
    return [row for row in rows if isinstance(row, Mapping)]


def _slot_label_table(path: Path, *, spec: SlotLabelSpec) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Stage B slot label table not found: {path}")
    frame = pd.read_parquet(path)
    missing = sorted({"id", spec.label_name, spec.target_family_count_column} - set(frame.columns))
    if missing:
        if spec.target_family_count_column in missing:
            raise ValueError(
                "Stage B slot diagnostics missing target-family count column "
                f"{spec.target_family_count_column!r} in {path}"
            )
        raise ValueError(f"Stage B slot label table missing column(s): {missing}")
    out = frame.loc[:, ["id", spec.label_name, spec.target_family_count_column]].copy()
    out["id"] = out["id"].astype(str)
    if out["id"].duplicated().any():
        duplicates = out.loc[out["id"].duplicated(), "id"].head(5).tolist()
        raise ValueError(f"Stage B slot label table contains duplicate id(s): {duplicates}")
    out[spec.label_name] = pd.to_numeric(out[spec.label_name], errors="raise").astype(float)
    out[spec.target_family_count_column] = pd.to_numeric(out[spec.target_family_count_column], errors="raise").astype(
        int
    )
    invalid_counts = out.loc[
        ~out[spec.target_family_count_column].between(0, spec.max_target_family_count),
        spec.target_family_count_column,
    ]
    if not invalid_counts.empty:
        raise ValueError(
            "Stage B slot diagnostics found invalid target-family count(s): "
            f"{sorted(set(map(int, invalid_counts.tolist())))}"
        )
    return out


def _count_distribution_rows(
    *,
    campaign: Mapping[str, Any],
    spec: SlotLabelSpec,
    label_name: str,
    round_index: int,
    selected: pd.DataFrame,
    pool_rows: pd.DataFrame,
) -> list[dict[str, Any]]:
    selected_counts = selected[spec.target_family_count_column].value_counts().to_dict()
    rows: list[dict[str, Any]] = []
    for _, pool_row in pool_rows.iterrows():
        target_count = int(pool_row["target_count"])
        selected_count = int(selected_counts.get(target_count, 0))
        rows.append(
            {
                "campaign_key": str(campaign["campaign_key"]),
                "label_name": label_name,
                "oracle_role": str(campaign["oracle_role"]),
                "split_id": str(campaign["split_id"]),
                "seed": int(campaign["seed"]),
                "round": int(round_index),
                "target_family_count_column": spec.target_family_count_column,
                "target_count": target_count,
                "is_deterministic_count": bool(target_count in spec.deterministic_counts),
                "selected_count": selected_count,
                "selected_fraction": selected_count / float(len(selected)) if len(selected) else np.nan,
                "pool_count": int(pool_row["pool_count"]),
                "pool_fraction": float(pool_row["pool_fraction"]),
                "pool_label_baseline_for_count": float(pool_row["pool_label_baseline_for_count"]),
            }
        )
    return rows


def _pool_count_distribution(frame: pd.DataFrame, *, spec: SlotLabelSpec) -> pd.DataFrame:
    grouped = (
        frame.groupby(spec.target_family_count_column, dropna=False)[spec.label_name]
        .agg(pool_count="count", pool_label_baseline_for_count="mean")
        .reset_index()
        .rename(columns={spec.target_family_count_column: "target_count"})
    )
    grouped["pool_fraction"] = grouped["pool_count"] / float(len(frame))
    all_counts = pd.DataFrame({"target_count": list(range(spec.max_target_family_count + 1))})
    out = all_counts.merge(grouped, on="target_count", how="left")
    out["pool_count"] = out["pool_count"].fillna(0).astype(int)
    out["pool_fraction"] = out["pool_fraction"].fillna(0.0)
    out["pool_label_baseline_for_count"] = out["pool_label_baseline_for_count"].fillna(np.nan)
    return out


def _baseline_by_count(frame: pd.DataFrame, *, spec: SlotLabelSpec) -> dict[int, float]:
    return {
        int(count): float(value)
        for count, value in frame.groupby(spec.target_family_count_column)[spec.label_name].mean().to_dict().items()
    }


def _nondeterministic_frame(frame: pd.DataFrame, *, spec: SlotLabelSpec) -> pd.DataFrame:
    return frame.loc[~_deterministic_mask(frame, spec=spec)].copy()


def _deterministic_mask(frame: pd.DataFrame, *, spec: SlotLabelSpec) -> pd.Series:
    return frame[spec.target_family_count_column].astype(int).isin(spec.deterministic_counts)


def _selected_count_stratified_expected(
    selected_nondeterministic: pd.DataFrame,
    *,
    spec: SlotLabelSpec,
    baseline_by_count: Mapping[int, float],
) -> float:
    if selected_nondeterministic.empty:
        return np.nan
    expected = [
        float(baseline_by_count[int(count)])
        for count in selected_nondeterministic[spec.target_family_count_column].astype(int).tolist()
    ]
    return float(np.mean(expected))


def _campaign_trajectory(trajectory: pd.DataFrame, campaign_key: str) -> pd.DataFrame:
    out = trajectory.loc[trajectory["campaign_key"] == campaign_key].copy()
    if out.empty:
        raise ValueError(f"missing slot diagnostic trajectory rows for campaign {campaign_key}")
    return out.sort_values("round")


def _paired_auc_delta(
    frame: pd.DataFrame,
    *,
    positive_column: str,
    null_column: str,
) -> tuple[float, float, float, int]:
    required = {"round", positive_column, null_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Stage B slot AUC missing column(s): {missing}")
    work = frame.loc[:, ["round", positive_column, null_column]].copy()
    work["round"] = pd.to_numeric(work["round"], errors="raise")
    work[positive_column] = pd.to_numeric(work[positive_column], errors="coerce")
    work[null_column] = pd.to_numeric(work[null_column], errors="coerce")
    work = work.loc[work[positive_column].map(_is_finite) & work[null_column].map(_is_finite)].sort_values("round")
    if len(work) < 2:
        return np.nan, np.nan, np.nan, int(len(work))
    rounds = work["round"].to_numpy(dtype=float)
    span = float(rounds[-1] - rounds[0])
    if span <= 0:
        raise ValueError("Stage B slot AUC rounds must increase")
    positive_auc = _normalized_trapezoid_auc(rounds, work[positive_column].to_numpy(dtype=float))
    null_auc = _normalized_trapezoid_auc(rounds, work[null_column].to_numpy(dtype=float))
    return positive_auc, null_auc, positive_auc - null_auc, int(len(work))


def _normalized_trapezoid_auc(rounds: np.ndarray, values: np.ndarray) -> float:
    span = float(rounds[-1] - rounds[0])
    widths = np.diff(rounds)
    area = np.sum(widths * (values[:-1] + values[1:]) / 2.0)
    return float(area / span)


def _slot_diagnostic_status(*, final_delta: float, auc_delta: float, valid_round_count: int) -> str:
    if valid_round_count < 2 or not _is_finite(final_delta) or not _is_finite(auc_delta):
        return INSUFFICIENT_NONDETERMINISTIC_SELECTION
    if final_delta > 0 and auc_delta > 0:
        return POSITION_SIGNAL_AFTER_COUNT_RESTRICTION
    return NOT_SEPARATED_AFTER_COUNT_RESTRICTION


def _selection_table(workdir: Path, *, round_index: int) -> pd.DataFrame:
    path = workdir / "outputs" / "rounds" / f"round_{int(round_index)}" / "selection" / "selection_top_k.csv"
    if not path.exists():
        raise FileNotFoundError(f"Stage B slot diagnostics selection artifact missing: {path}")
    frame = pd.read_csv(path)
    if "id" not in frame.columns:
        raise ValueError(f"Stage B slot diagnostics selection artifact missing id column: {path}")
    frame["id"] = frame["id"].astype(str)
    return frame


def _campaign_workdir(config_path: Path) -> Path:
    if config_path.name != "campaign.yaml" or config_path.parent.name != "configs":
        raise ValueError(f"Stage B config path does not follow campaign/configs/campaign.yaml layout: {config_path}")
    return config_path.parent.parent


def _reject_duplicate_ids(ids: Sequence[str], *, path: Path, round_index: int) -> None:
    if len(set(ids)) != len(ids):
        raise ValueError(
            f"Stage B slot diagnostics selection artifact has duplicate id(s): {path}, round={round_index}"
        )


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not _is_finite(numerator) or not _is_finite(denominator) or denominator <= 0:
        return np.nan
    return float(numerator / denominator)


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Stage B slot diagnostics config manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Stage B slot diagnostics config manifest must be a JSON object: {path}")
    return payload
