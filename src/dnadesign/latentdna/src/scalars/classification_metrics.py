"""Binary classification metric helpers for scalar scorecards."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from ..contracts.errors import ContractViolationError


def _binary_label_array(labels: np.ndarray) -> np.ndarray:
    values = np.asarray(labels)
    if values.ndim != 1:
        raise ContractViolationError("binary metrics require one-dimensional labels")
    invalid = sorted(set(values.tolist()) - {0, 1})
    if invalid:
        raise ContractViolationError(f"binary metrics require labels encoded as 0/1, observed {invalid!r}")
    return values.astype(np.int8, copy=False)


def _score_array(scores: np.ndarray, *, expected_shape: tuple[int, ...]) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != expected_shape:
        raise ContractViolationError("binary metrics require labels and scores to share one shape")
    if values.ndim != 1:
        raise ContractViolationError("binary metrics require one-dimensional scores")
    if not np.all(np.isfinite(values)):
        raise ContractViolationError("binary metrics require finite score values")
    return values


def _validated_binary_inputs(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    label_values = _binary_label_array(labels)
    score_values = _score_array(scores, expected_shape=label_values.shape)
    return label_values, score_values


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Return threshold-grouped average precision for binary 0/1 labels."""

    label_values, score_values = _validated_binary_inputs(labels, scores)
    positives = int(np.sum(label_values))
    if positives == 0:
        return float("nan")
    order = np.argsort(score_values, kind="stable")[::-1]
    sorted_scores = score_values[order]
    sorted_labels = label_values[order]
    true_positives = 0
    previous_recall = 0.0
    area = 0.0
    start = 0
    while start < sorted_labels.size:
        end = start + 1
        while end < sorted_labels.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        true_positives += int(np.sum(sorted_labels[start:end]))
        recall = true_positives / float(positives)
        precision = true_positives / float(end)
        area += (recall - previous_recall) * precision
        previous_recall = recall
        start = end
    return float(area)


def _average_tie_ranks(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < sorted_scores.size:
        end = start + 1
        while end < sorted_scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Return AUROC with tied scores contributing half credit."""

    label_values, score_values = _validated_binary_inputs(labels, scores)
    positives = int(np.sum(label_values))
    negatives = int(label_values.size - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = _average_tie_ranks(score_values)
    positive_rank_sum = float(np.sum(ranks[label_values == 1], dtype=np.float64))
    return float((positive_rank_sum - (positives * (positives + 1) / 2.0)) / (positives * negatives))


def binary_metrics(
    *,
    rows: Iterable[dict[str, Any]],
    label_column: str,
    positive_values: set[str],
    score_values: np.ndarray,
) -> dict[str, float]:
    labels = np.asarray([1 if str(row.get(label_column)) in positive_values else 0 for row in rows], dtype=np.int8)
    return {
        "auroc": roc_auc(labels, score_values),
        "auprc": average_precision(labels, score_values),
    }


def dual_joint_margin(left_scores: np.ndarray, right_scores: np.ndarray) -> np.ndarray:
    left = np.asarray(left_scores, dtype=np.float64)
    right = np.asarray(right_scores, dtype=np.float64)
    if left.shape != right.shape:
        raise ContractViolationError("dual joint margin requires aligned score vectors")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ContractViolationError("dual joint margin requires finite score values")
    return np.minimum(left, right)
