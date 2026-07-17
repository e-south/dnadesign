"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/runtime/multistate_behavior_prediction.py

Verify one fixed OPAL prediction run for behavior shadow scoring.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from dnadesign.opal import score_response_magnitude_feasibility

from .multistate_behavior_run_contract import RESPONSE_Y_COLUMNS, verify_behavior_run_receipt
from .publication import sha256_file


@dataclass(frozen=True)
class VerifiedBehaviorPredictionRun:
    """Fixed prediction matrix, comparator scores, and run-scoped receipt."""

    predictions: pd.DataFrame
    comparator_scores: pd.DataFrame
    source: dict[str, object]


def load_verified_behavior_prediction_run(
    *,
    campaign_dir: Path,
    candidate_records_path: Path,
    prediction_run_id: str,
    state_ids: tuple[str, ...],
    target_masks: Mapping[str, tuple[float, ...]],
    comparator_calibration_by_view: Mapping[str, Mapping[str, float]],
    comparator_objective_name: str,
    comparator_channel: str,
    comparator_direction: str,
    model_name: str,
    model_params: Mapping[str, object],
    raw_top_k: int,
) -> VerifiedBehaviorPredictionRun:
    """Bind scoring to the exact eligible pool recorded by one OPAL run."""

    campaign_root = Path(campaign_dir).resolve()
    prediction_parts = sorted((campaign_root / "outputs/ledger/predictions").glob("*.parquet"))
    if not prediction_parts:
        raise FileNotFoundError("OPAL prediction ledger is missing.")
    rows, prediction_inventory = _rows_for_run(
        prediction_parts,
        run_id=prediction_run_id,
        campaign_dir=campaign_root,
        context="prediction ledger",
    )
    receipt, receipt_inventory = _load_run_receipt(
        campaign_dir=campaign_root,
        run_id=prediction_run_id,
    )
    if rows["id"].astype(str).duplicated().any():
        raise ValueError("declared prediction run contains duplicate candidate ids.")
    run_lineage = verify_behavior_run_receipt(
        receipt,
        rows=rows,
        prediction_run_id=prediction_run_id,
        state_ids=state_ids,
        target_masks=target_masks,
        comparator_calibration_by_view=comparator_calibration_by_view,
        comparator_objective_name=comparator_objective_name,
        comparator_channel=comparator_channel,
        comparator_direction=comparator_direction,
        model_name=model_name,
        model_params=model_params,
        raw_top_k=raw_top_k,
    )
    candidate_projection = _verify_prediction_candidates(rows, candidate_records_path=Path(candidate_records_path))
    values = np.vstack(rows["pred__y_hat_model"].to_numpy()).astype(float)
    if values.shape != (len(rows), len(RESPONSE_Y_COLUMNS)) or not np.isfinite(values).all():
        raise ValueError("prediction run must contain one finite eight-value prediction per candidate.")
    if not rows["pred__y_dim"].eq(len(RESPONSE_Y_COLUMNS)).all():
        raise ValueError("prediction run y-dimension receipt disagrees with its prediction vectors.")

    file_inventory = {
        "prediction_parts": prediction_inventory,
        "run_receipt_parts": receipt_inventory,
    }
    source_sha = (
        "sha256:"
        + hashlib.sha256(json.dumps(file_inventory, separators=(",", ":"), sort_keys=True).encode("utf-8")).hexdigest()
    )
    predictions = pd.DataFrame(values, columns=RESPONSE_Y_COLUMNS)
    predictions.insert(0, "id", rows["id"].astype(str).to_numpy())
    predictions["prediction_run_id"] = prediction_run_id
    predictions["prediction_source_sha256"] = source_sha
    comparator_scores = _comparator_scores(
        rows,
        values=values,
        target_masks=target_masks,
        calibration_by_view=comparator_calibration_by_view,
        comparator_channel=comparator_channel,
        prediction_run_id=prediction_run_id,
        source_sha=source_sha,
    )
    return VerifiedBehaviorPredictionRun(
        predictions=predictions,
        comparator_scores=comparator_scores,
        source={
            "run_id": prediction_run_id,
            "ledger_root": "outputs/ledger",
            "ledger_sha256": source_sha,
            "files": file_inventory,
            "candidate_count": len(rows),
            "run_receipt_scored_count": int(receipt["stats__n_scored"]),
            "run_lineage": run_lineage,
            "candidate_projection": candidate_projection,
        },
    )


def _comparator_scores(
    rows: pd.DataFrame,
    *,
    values: np.ndarray,
    target_masks: Mapping[str, tuple[float, ...]],
    calibration_by_view: Mapping[str, Mapping[str, float]],
    comparator_channel: str,
    prediction_run_id: str,
    source_sha: str,
) -> pd.DataFrame:
    replayed: dict[str, dict[str, np.ndarray]] = {}
    for view_id, target_mask in target_masks.items():
        score = score_response_magnitude_feasibility(
            values,
            target_mask=target_mask,
            calibration=calibration_by_view[view_id],
        )
        replayed[view_id] = {
            "feasibility_margin": score.feasibility_margin,
            "response_separation": score.components.response_separation,
            "on_magnitude_floor": score.components.on_magnitude_floor,
            "off_magnitude_ceiling": score.components.off_magnitude_ceiling,
        }
    expected_names = {f"{view_id}/{channel}" for view_id in target_masks for channel in next(iter(replayed.values()))}
    records: list[dict[str, object]] = []
    for row_index, row in enumerate(rows.itertuples(index=False)):
        channels: dict[str, float] = {}
        for item in row.pred__score_channels:
            if not isinstance(item, dict) or set(item) != {"name", "value"}:
                raise ValueError("prediction score channels must be name/value mappings.")
            name = item["name"]
            value = item["value"]
            if not isinstance(name, str) or not name or name in channels:
                raise ValueError("prediction score channel names must be nonempty and unique per candidate.")
            if isinstance(value, bool) or not np.isfinite(float(value)):
                raise ValueError(f"prediction score channel {name!r} must be finite numeric evidence.")
            channels[name] = float(value)
        if set(channels) != expected_names:
            raise ValueError("prediction score channels do not match the receipted comparator outputs.")
        for view_id in target_masks:
            for channel, expected_values in replayed[view_id].items():
                name = f"{view_id}/{channel}"
                if not np.isclose(channels[name], float(expected_values[row_index]), rtol=1.0e-12, atol=1.0e-12):
                    raise ValueError(f"prediction score channel {name!r} does not replay from the prediction vector.")
            records.append(
                {
                    "id": str(row.id),
                    "selection_view_id": view_id,
                    "hard_score": float(replayed[view_id][comparator_channel][row_index]),
                    "prediction_run_id": prediction_run_id,
                    "prediction_source_sha256": source_sha,
                }
            )
    return pd.DataFrame.from_records(records)


def _load_run_receipt(*, campaign_dir: Path, run_id: str) -> tuple[pd.Series, list[dict[str, object]]]:
    parts = sorted((campaign_dir / "outputs/ledger/runs.parquet").glob("*.parquet"))
    if not parts:
        raise FileNotFoundError("OPAL run ledger is missing.")
    rows, inventory = _rows_for_run(parts, run_id=run_id, campaign_dir=campaign_dir, context="run ledger")
    if len(rows) != 1:
        raise ValueError(f"prediction run must have exactly one run receipt; observed {len(rows)}.")
    return rows.iloc[0], inventory


def _rows_for_run(
    files: list[Path],
    *,
    run_id: str,
    campaign_dir: Path,
    context: str,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    frames: list[pd.DataFrame] = []
    inventory: list[dict[str, object]] = []
    for path in files:
        frame = pd.read_parquet(path)
        if "run_id" not in frame:
            raise ValueError(f"{context} part lacks run_id: {path}")
        matching = frame.loc[frame["run_id"].astype(str).eq(run_id)].copy()
        if matching.empty:
            continue
        frames.append(matching)
        inventory.append(_portable_file_receipt(path, campaign_dir=campaign_dir))
    if not frames:
        raise ValueError(f"declared prediction run is absent from the {context}: {run_id}")
    return pd.concat(frames, ignore_index=True), inventory


def _verify_prediction_candidates(rows: pd.DataFrame, *, candidate_records_path: Path) -> dict[str, object]:
    if missing := sorted({"id", "sequence"} - set(rows.columns)):
        raise ValueError(f"prediction rows lack candidate identity fields: {missing}")
    candidates = pd.read_parquet(candidate_records_path, columns=["id", "sequence"])
    if candidates["id"].astype(str).duplicated().any():
        raise ValueError("campaign candidate records contain duplicate ids.")
    source = candidates.assign(id=candidates["id"].astype(str), sequence=candidates["sequence"].astype(str))
    observed = rows.loc[:, ["id", "sequence"]].assign(
        id=rows["id"].astype(str),
        sequence=rows["sequence"].astype(str),
    )
    aligned = observed.merge(source, on="id", how="left", suffixes=("_prediction", "_candidate"), validate="one_to_one")
    if aligned["sequence_candidate"].isna().any():
        raise ValueError("prediction run contains ids absent from campaign candidate records.")
    if not aligned["sequence_prediction"].eq(aligned["sequence_candidate"]).all():
        raise ValueError("prediction sequence identity disagrees with campaign candidate records.")
    projection = (
        aligned.loc[:, ["id", "sequence_candidate"]]
        .rename(columns={"sequence_candidate": "sequence"})
        .sort_values("id", kind="mergesort")
    )
    rendered = projection.to_json(orient="records")
    return {
        "source_row_count": len(candidates),
        "scored_row_count": len(projection),
        "sha256": "sha256:" + hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
    }


def _portable_file_receipt(path: Path, *, campaign_dir: Path) -> dict[str, object]:
    resolved = path.resolve()
    if not resolved.is_relative_to(campaign_dir):
        raise ValueError(f"OPAL ledger artifact escapes the campaign directory: {resolved}")
    return {
        "path": resolved.relative_to(campaign_dir).as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


__all__ = ["VerifiedBehaviorPredictionRun", "load_verified_behavior_prediction_run"]
