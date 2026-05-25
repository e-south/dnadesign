"""Prediction-ledger evaluation for the DenseGen axis OPAL probe."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .artifacts import RunSpec
from .axis_oracle import _ok_labels
from .constants import AXIS_CLASS_TO_LOGIC4
from .decision import _binomial_tail_ge
from .prediction_ledger import prediction_id_problems, read_probe_predictions
from .prediction_scoring import (
    label_lookup,
    macro_f1,
    predicted_axis_classes,
    selected_bool_mask,
    top_ids_from_prediction_frame,
    validate_prediction_selection_contract,
)


def _evaluate_run(
    *,
    run: RunSpec,
    positive_labels: pd.DataFrame,
    run_labels: pd.DataFrame,
    split_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        predictions = read_probe_predictions(run.config_path)
    except Exception as exc:
        raise RuntimeError(f"failed to read OPAL predictions for scored run {run.run_key}: {exc}") from exc
    train_ids = set(map(str, split_metadata["train_ids"]))
    return _evaluate_prediction_frame(
        run=run,
        predictions=predictions,
        positive_labels=positive_labels,
        run_labels=run_labels,
        split_metadata=split_metadata,
        train_ids=train_ids,
    )


def _evaluate_run_rounds(
    *,
    run: RunSpec,
    positive_labels: pd.DataFrame,
    run_labels: pd.DataFrame,
    split_metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    try:
        predictions = read_probe_predictions(run.config_path, round_selector="all")
    except Exception as exc:
        raise RuntimeError(f"failed to read OPAL predictions for scored run {run.run_key}: {exc}") from exc
    if predictions.empty:
        raise RuntimeError(f"missing OPAL prediction artifacts for scored run {run.run_key}: {run.workdir}")
    missing_group_columns = sorted({"run_id", "as_of_round"} - set(predictions.columns))
    if missing_group_columns:
        raise RuntimeError(
            f"OPAL prediction artifacts for scored run {run.run_key} missing column(s): {missing_group_columns}"
        )

    rows: list[dict[str, Any]] = []
    initial_train_ids = set(map(str, split_metadata["train_ids"]))
    for (round_index, _run_id), frame in predictions.groupby(["as_of_round", "run_id"], sort=True, dropna=False):
        if pd.isna(round_index):
            raise RuntimeError(f"OPAL prediction artifacts for scored run {run.run_key} contain null as_of_round")
        train_ids = _labeled_ids_for_round(
            run.sidecar_path, initial_train_ids=initial_train_ids, round_index=int(round_index)
        )
        row = _evaluate_prediction_frame(
            run=run,
            predictions=frame.copy(),
            positive_labels=positive_labels,
            run_labels=run_labels,
            split_metadata=split_metadata,
            train_ids=train_ids,
        )
        row["metric_scope"] = "round"
        rows.append(row)
    return rows


def _labeled_ids_for_round(sidecar_path: Path, *, initial_train_ids: set[str], round_index: int) -> set[str]:
    labeled = set(initial_train_ids)
    if not sidecar_path.exists():
        return labeled
    try:
        frame = pd.read_parquet(sidecar_path, columns=["id", "observed_round"])
    except Exception as exc:
        raise RuntimeError(f"observed-label sidecar unreadable for round metric evaluation: {sidecar_path}") from exc
    if frame.empty:
        return labeled
    observed_round = pd.to_numeric(frame["observed_round"], errors="coerce")
    if observed_round.isna().any():
        raise RuntimeError(f"observed-label sidecar contains nonnumeric observed_round values: {sidecar_path}")
    labeled.update(frame.loc[observed_round <= int(round_index), "id"].dropna().astype(str).tolist())
    return labeled


def _evaluate_prediction_frame(
    *,
    run: RunSpec,
    predictions: pd.DataFrame,
    positive_labels: pd.DataFrame,
    run_labels: pd.DataFrame,
    split_metadata: Mapping[str, Any],
    train_ids: set[str],
) -> dict[str, Any]:
    true_labels = label_lookup(positive_labels)
    oracle_labels = label_lookup(run_labels)
    if "eval_ids" in split_metadata:
        eval_ids = set(map(str, split_metadata["eval_ids"])) - train_ids
    elif run.split_id == "leave_sigma35_variant":
        raise RuntimeError(f"split metadata for scored run {run.run_key} missing eval_ids")
    else:
        eval_ids = set(_ok_labels(positive_labels)["id"].astype(str).tolist()) - train_ids

    row: dict[str, Any] = {
        "run_key": run.run_key,
        "campaign": run.campaign_key,
        "oracle_id": run.oracle_id,
        "split_id": run.split_id,
        "seed": run.seed,
        "label_family_id": run.label_family_id,
        "target_class": run.target_class,
        "train_count": int(len(train_ids)),
        "eval_count": int(len(eval_ids)),
    }
    _validate_prediction_scope(run=run, predictions=predictions, row=row)
    prediction_problems = prediction_id_problems(predictions, eval_ids, run_key=run.run_key)
    if prediction_problems:
        raise RuntimeError(f"OPAL prediction artifacts invalid: {'; '.join(prediction_problems)}")
    validate_prediction_selection_contract(predictions)

    frame = predictions.loc[predictions["id"].astype(str).isin(eval_ids)].copy()
    frame["pred_axis_class"] = predicted_axis_classes(frame["pred__y_hat_model"].tolist())
    frame["true_axis_class"] = frame["id"].astype(str).map(true_labels)
    frame["oracle_axis_class"] = frame["id"].astype(str).map(oracle_labels)
    row["axis4_macro_f1_true"] = macro_f1(frame["true_axis_class"].astype(str), frame["pred_axis_class"].astype(str))
    row["axis4_macro_f1_oracle"] = macro_f1(
        frame["oracle_axis_class"].astype(str), frame["pred_axis_class"].astype(str)
    )
    _append_selection_metrics(row=row, run=run, frame=frame, true_labels=true_labels, oracle_labels=oracle_labels)
    return row


def _validate_prediction_scope(*, run: RunSpec, predictions: pd.DataFrame, row: dict[str, Any]) -> None:
    if predictions.empty:
        raise RuntimeError(f"missing OPAL prediction artifacts for scored run {run.run_key}: {run.workdir}")
    missing = sorted(
        {"id", "pred__y_hat_model", "pred__score_selected", "sel__is_selected", "sel__rank_competition"}
        - set(predictions.columns)
    )
    if missing:
        raise RuntimeError(f"OPAL prediction artifacts for scored run {run.run_key} missing column(s): {missing}")
    run_ids = sorted({str(value) for value in predictions["run_id"].dropna().tolist()})
    rounds = sorted({int(value) for value in predictions["as_of_round"].dropna().tolist()})
    if len(run_ids) != 1:
        raise RuntimeError(f"OPAL prediction artifacts for scored run {run.run_key} must contain one run_id")
    if len(rounds) != 1:
        raise RuntimeError(f"OPAL prediction artifacts for scored run {run.run_key} must contain one as_of_round")
    row["run_id"] = run_ids[0]
    row["as_of_round"] = rounds[0]


def _append_selection_metrics(
    *,
    row: dict[str, Any],
    run: RunSpec,
    frame: pd.DataFrame,
    true_labels: pd.Series,
    oracle_labels: pd.Series,
) -> None:
    selection_k = int(run.selection_k)
    selected_mask = selected_bool_mask(frame)
    row["selected_count_in_eval"] = int(selected_mask.sum())
    row["selection_k"] = selection_k
    if int(row["selected_count_in_eval"]) != selection_k:
        raise RuntimeError(
            f"scored run {run.run_key} expected {selection_k} evaluable selected id(s) "
            f"inside split {run.split_id}, got {row['selected_count_in_eval']}."
        )
    selected_ids = top_ids_from_prediction_frame(frame, k=selection_k)
    if len(selected_ids) != selection_k:
        raise RuntimeError(
            f"scored run {run.run_key} could not resolve {selection_k} selected id(s) "
            f"inside split {run.split_id}, got {len(selected_ids)}."
        )
    selected_true = true_labels.reindex(selected_ids).astype(str).tolist()
    selected_oracle = oracle_labels.reindex(selected_ids).astype(str).tolist()
    true_eval_classes = frame["true_axis_class"].astype(str).tolist()
    oracle_eval_classes = frame["oracle_axis_class"].astype(str).tolist()
    selected_target_count_true = int(sum(axis_class == run.target_class for axis_class in selected_true))
    selected_target_count_oracle = int(sum(axis_class == run.target_class for axis_class in selected_oracle))
    prev_true, prec_true, lift_true = _lift(run.target_class, true_eval_classes, selected_true)
    prev_oracle, prec_oracle, lift_oracle = _lift(run.target_class, oracle_eval_classes, selected_oracle)
    row.update(
        {
            "selected_ids": selected_ids,
            "selected_target_count_true": selected_target_count_true,
            "selected_target_count_oracle": selected_target_count_oracle,
            "selected_target_count_label_true": f"{selected_target_count_true}/{selection_k}",
            "selected_target_count_label_oracle": f"{selected_target_count_oracle}/{selection_k}",
            "target_class_prevalence_true": prev_true,
            "selected_target_precision_at_k_true": prec_true,
            "target_lift_at_k_true": lift_true,
            "selected_target_binomial_tail_p_true": _binomial_tail_ge(
                selected_target_count_true, selection_k, prev_true
            ),
            "target_class_prevalence_oracle": prev_oracle,
            "selected_target_precision_at_k_oracle": prec_oracle,
            "target_lift_at_k_oracle": lift_oracle,
            "selected_target_binomial_tail_p_oracle": _binomial_tail_ge(
                selected_target_count_oracle, selection_k, prev_oracle
            ),
            "off_target_class_distribution_true": {
                axis_class: int(selected_true.count(axis_class)) for axis_class in AXIS_CLASS_TO_LOGIC4
            },
        }
    )


def _lift(target_class: str, classes: Sequence[str], selected_classes: Sequence[str]) -> tuple[float, float, float]:
    if not classes or not selected_classes:
        return float("nan"), float("nan"), float("nan")
    prevalence = float(pd.Series(classes).eq(target_class).mean())
    precision = float(pd.Series(selected_classes).eq(target_class).mean())
    return prevalence, precision, float("nan") if prevalence == 0.0 else precision / prevalence
