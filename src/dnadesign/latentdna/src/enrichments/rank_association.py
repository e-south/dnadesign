"""Rank association tests for categorical features over numeric axes."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
from scipy.stats import mannwhitneyu

from ..contracts.errors import ContractViolationError
from .enrichment_stats import benjamini_hochberg
from .table_contracts import require_sequence, string_values

_CONTRACT_NAME = "rank_association"
_SUPPORTED_ALTERNATIVES = frozenset({"greater", "less", "two-sided"})
_SUPPORTED_P_VALUE_METHOD = "scipy_mannwhitneyu_asymptotic"
_SUPPORTED_FDR_METHOD = "benjamini_hochberg"


@dataclass(frozen=True, slots=True)
class RankAssociationConfig:
    """Statistical thresholds and methods for feature rank tests."""

    min_feature_support: int
    alternative: str = "greater"
    p_value_method: str = _SUPPORTED_P_VALUE_METHOD
    fdr_method: str = _SUPPORTED_FDR_METHOD


def _unique_ids(values: Iterable[str], *, field_name: str) -> list[str]:
    ids = [
        str(value).strip() for value in require_sequence(values, field_name=field_name, contract_name=_CONTRACT_NAME)
    ]
    blank = [index for index, value in enumerate(ids) if not value]
    if blank:
        raise ContractViolationError(f"{_CONTRACT_NAME} {field_name} contains blank identifiers")
    duplicates = sorted(value for value, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ContractViolationError(f"{_CONTRACT_NAME} {field_name} contains duplicate identifiers: {duplicates[:5]}")
    return ids


def _validate_config(config: RankAssociationConfig) -> None:
    if config.min_feature_support < 1:
        raise ContractViolationError(f"{_CONTRACT_NAME} min_feature_support must be >= 1")
    if config.alternative not in _SUPPORTED_ALTERNATIVES:
        raise ContractViolationError(f"{_CONTRACT_NAME} alternative must be one of {sorted(_SUPPORTED_ALTERNATIVES)}")
    if config.p_value_method != _SUPPORTED_P_VALUE_METHOD:
        raise ContractViolationError(f"{_CONTRACT_NAME} only supports p_value_method={_SUPPORTED_P_VALUE_METHOD!r}")
    if config.fdr_method != _SUPPORTED_FDR_METHOD:
        raise ContractViolationError(f"{_CONTRACT_NAME} only supports fdr_method={_SUPPORTED_FDR_METHOD!r}")


def _score_matrix(
    *,
    universe: set[str],
    score_by_subject: Mapping[str, Mapping[str, object]],
    axis_ids: list[str],
) -> dict[str, dict[str, float]]:
    if not isinstance(score_by_subject, Mapping):
        raise ContractViolationError(f"{_CONTRACT_NAME} score_by_subject must be a mapping")
    subject_ids = {str(subject).strip() for subject in score_by_subject}
    outside = sorted(subject_ids.difference(universe))
    if outside:
        raise ContractViolationError(f"{_CONTRACT_NAME} scores include subjects outside universe: {outside[:5]}")
    missing_subjects = sorted(universe.difference(subject_ids))
    if missing_subjects:
        raise ContractViolationError(f"{_CONTRACT_NAME} scores are missing subjects: {missing_subjects[:5]}")

    scores: dict[str, dict[str, float]] = {}
    for subject, raw_scores in score_by_subject.items():
        subject_id = str(subject).strip()
        if not isinstance(raw_scores, Mapping):
            raise ContractViolationError(f"{_CONTRACT_NAME} scores for subject {subject_id!r} must be a mapping")
        subject_scores: dict[str, float] = {}
        for axis in axis_ids:
            if axis not in raw_scores:
                raise ContractViolationError(
                    f"{_CONTRACT_NAME} scores for subject {subject_id!r} missing axis {axis!r}"
                )
            try:
                score = float(raw_scores[axis])
            except (TypeError, ValueError) as exc:
                raise ContractViolationError(
                    f"{_CONTRACT_NAME} score for subject {subject_id!r} axis {axis!r} is not numeric"
                ) from exc
            if not math.isfinite(score):
                raise ContractViolationError(
                    f"{_CONTRACT_NAME} score for subject {subject_id!r} axis {axis!r} is not finite"
                )
            subject_scores[axis] = score
        scores[subject_id] = subject_scores
    return scores


def _feature_members(
    *,
    universe: set[str],
    features_by_subject: Mapping[str, Iterable[str]],
    feature_labels: Mapping[str, str],
) -> tuple[dict[str, set[str]], dict[str, str]]:
    if not isinstance(features_by_subject, Mapping):
        raise ContractViolationError(f"{_CONTRACT_NAME} features_by_subject must be a mapping")
    if not isinstance(feature_labels, Mapping):
        raise ContractViolationError(f"{_CONTRACT_NAME} feature_labels must be a mapping")
    labels = {str(key).strip(): str(value).strip() for key, value in feature_labels.items()}
    if not labels:
        raise ContractViolationError(f"{_CONTRACT_NAME} requires at least one feature label")
    blank_features = sorted(key for key, value in labels.items() if not key or not value)
    if blank_features:
        raise ContractViolationError(f"{_CONTRACT_NAME} feature labels contain blank keys or values")

    members_by_feature: dict[str, set[str]] = {feature: set() for feature in labels}
    for subject, raw_features in features_by_subject.items():
        subject_id = str(subject).strip()
        if subject_id not in universe:
            raise ContractViolationError(f"{_CONTRACT_NAME} feature subject {subject_id!r} is outside universe")
        features = require_sequence(
            raw_features,
            field_name=f"features for subject {subject_id!r}",
            contract_name=_CONTRACT_NAME,
        )
        for raw_feature in features:
            feature = str(raw_feature).strip()
            if not feature:
                raise ContractViolationError(f"{_CONTRACT_NAME} feature membership contains a blank feature")
            if feature not in labels:
                raise ContractViolationError(f"{_CONTRACT_NAME} feature {feature!r} is missing a display label")
            members_by_feature[feature].add(subject_id)
    if not any(members_by_feature.values()):
        raise ContractViolationError(f"{_CONTRACT_NAME} matched no feature memberships")
    return members_by_feature, labels


def _mann_whitney_row(
    *,
    positive_scores: list[float],
    negative_scores: list[float],
    config: RankAssociationConfig,
) -> tuple[float, float, float, float]:
    if not positive_scores or not negative_scores:
        return float("nan"), float("nan"), float("nan"), float("nan")
    result = mannwhitneyu(
        positive_scores,
        negative_scores,
        alternative=config.alternative,
        method="asymptotic",
        use_continuity=True,
    )
    u_statistic = float(result.statistic)
    auc = u_statistic / float(len(positive_scores) * len(negative_scores))
    rank_biserial = (2.0 * auc) - 1.0
    return u_statistic, auc, rank_biserial, float(result.pvalue)


def rank_association_rows(
    *,
    universe_ids: Iterable[str],
    score_by_subject: Mapping[str, Mapping[str, object]],
    features_by_subject: Mapping[str, Iterable[str]],
    feature_labels: Mapping[str, str],
    axis_ids: Iterable[str],
    config: RankAssociationConfig,
    common_features: Iterable[str],
) -> list[dict[str, object]]:
    """Test whether feature-positive subjects rank higher on numeric axes.

    The primitive is domain-neutral: callers decide whether features are
    regulators, motifs, labels, ontology terms, or other categorical sidecars.
    """

    _validate_config(config)
    subject_ids = _unique_ids(universe_ids, field_name="universe_ids")
    universe = set(subject_ids)
    axes = string_values(axis_ids, field_name="axis_ids", contract_name=_CONTRACT_NAME)
    if not axes:
        raise ContractViolationError(f"{_CONTRACT_NAME} requires at least one axis")
    duplicates = sorted(axis for axis, count in Counter(axes).items() if count > 1)
    if duplicates:
        raise ContractViolationError(f"{_CONTRACT_NAME} axis_ids contains duplicate identifiers: {duplicates[:5]}")
    scores = _score_matrix(universe=universe, score_by_subject=score_by_subject, axis_ids=axes)
    members_by_feature, labels = _feature_members(
        universe=universe,
        features_by_subject=features_by_subject,
        feature_labels=feature_labels,
    )
    common = {
        str(value).strip().casefold()
        for value in require_sequence(common_features, field_name="common_features", contract_name=_CONTRACT_NAME)
        if str(value).strip()
    }

    feature_keys = sorted(labels, key=lambda key: labels[key].casefold())
    rows: list[dict[str, object]] = []
    p_values: list[float] = []
    for axis in axes:
        axis_scores = {subject: scores[subject][axis] for subject in subject_ids}
        for feature in feature_keys:
            positive_subjects = members_by_feature[feature]
            negative_subjects = universe.difference(positive_subjects)
            positive_scores = [axis_scores[subject] for subject in subject_ids if subject in positive_subjects]
            negative_scores = [axis_scores[subject] for subject in subject_ids if subject in negative_subjects]
            u_statistic, auc, rank_biserial, p_value = _mann_whitney_row(
                positive_scores=positive_scores,
                negative_scores=negative_scores,
                config=config,
            )
            p_values.append(p_value)
            passes_min_feature_support = len(positive_subjects) >= config.min_feature_support
            is_common_feature = feature.casefold() in common or labels[feature].casefold() in common
            notes = []
            if not passes_min_feature_support:
                notes.append("below_min_feature_support")
            if not positive_scores or not negative_scores:
                notes.append("degenerate_comparison")
            if is_common_feature:
                notes.append("common_feature")
            rows.append(
                {
                    "feature": labels[feature],
                    "axis": axis,
                    "n_total": len(subject_ids),
                    "n_with": len(positive_subjects),
                    "n_without": len(negative_subjects),
                    "median_with": float(np.median(positive_scores)) if positive_scores else float("nan"),
                    "median_without": float(np.median(negative_scores)) if negative_scores else float("nan"),
                    "u_statistic": u_statistic,
                    "auc": auc,
                    "rank_biserial": rank_biserial,
                    "p_value": p_value,
                    "q_value": float("nan"),
                    "p_value_method": config.p_value_method,
                    "p_value_alternative": config.alternative,
                    "fdr_method": config.fdr_method,
                    "passes_min_feature_support": passes_min_feature_support,
                    "is_common_feature": is_common_feature,
                    "notes": ";".join(notes),
                }
            )

    q_values = benjamini_hochberg(p_values)
    for row, q_value in zip(rows, q_values, strict=True):
        row["q_value"] = q_value
    return rows
