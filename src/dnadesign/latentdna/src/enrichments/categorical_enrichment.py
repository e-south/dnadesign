"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/enrichments/categorical_enrichment.py

Generic categorical-feature enrichment over explicit row groups.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from ..contracts.errors import ContractViolationError
from .enrichment_stats import benjamini_hochberg, hypergeometric_survival, odds_ratio

_CONTRACT_NAME = "categorical_enrichment"


@dataclass(frozen=True, slots=True)
class CategoricalEnrichmentConfig:
    """Statistical thresholds and methods for categorical enrichment."""

    min_feature_support: int
    min_group_hits: int
    p_value_method: str = "hypergeometric_survival"
    fdr_method: str = "benjamini_hochberg"


@dataclass(frozen=True, slots=True)
class CategoricalEnrichmentGroup:
    """A named subset of the universe to test for feature enrichment."""

    labels: Mapping[str, object]
    members: frozenset[str]


def _require_sequence(value: object, *, field_name: str) -> list[object]:
    if isinstance(value, str) or isinstance(value, Mapping) or not isinstance(value, Iterable):
        raise ContractViolationError(f"{_CONTRACT_NAME} {field_name} must be a sequence, not a scalar or mapping")
    return list(value)


def _unique_ids(values: Iterable[str], *, field_name: str) -> list[str]:
    ids = [str(value).strip() for value in _require_sequence(values, field_name=field_name)]
    blank = [index for index, value in enumerate(ids) if not value]
    if blank:
        raise ContractViolationError(f"{_CONTRACT_NAME} {field_name} contains blank identifiers")
    duplicates = sorted(value for value, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ContractViolationError(f"{_CONTRACT_NAME} {field_name} contains duplicate identifiers: {duplicates[:5]}")
    return ids


def _validate_config(config: CategoricalEnrichmentConfig) -> None:
    if config.min_feature_support < 1:
        raise ContractViolationError(f"{_CONTRACT_NAME} min_feature_support must be >= 1")
    if config.min_group_hits < 1:
        raise ContractViolationError(f"{_CONTRACT_NAME} min_group_hits must be >= 1")
    if config.p_value_method != "hypergeometric_survival":
        raise ContractViolationError(f"{_CONTRACT_NAME} only supports p_value_method='hypergeometric_survival'")
    if config.fdr_method != "benjamini_hochberg":
        raise ContractViolationError(f"{_CONTRACT_NAME} only supports fdr_method='benjamini_hochberg'")


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
        features = _require_sequence(raw_features, field_name=f"features for subject {subject_id!r}")
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


def _validated_groups(
    *,
    groups: Iterable[CategoricalEnrichmentGroup],
    universe: set[str],
) -> list[CategoricalEnrichmentGroup]:
    validated: list[CategoricalEnrichmentGroup] = []
    for group in _require_sequence(groups, field_name="groups"):
        if not isinstance(group, CategoricalEnrichmentGroup):
            raise ContractViolationError(f"{_CONTRACT_NAME} groups must contain CategoricalEnrichmentGroup values")
        if not isinstance(group.labels, Mapping):
            raise ContractViolationError(f"{_CONTRACT_NAME} group labels must be a mapping")
        labels = {str(key).strip(): value for key, value in group.labels.items()}
        if not labels or any(not key for key in labels):
            raise ContractViolationError(f"{_CONTRACT_NAME} group labels must not be empty")
        raw_members = _require_sequence(group.members, field_name="group.members")
        members = frozenset(str(value).strip() for value in raw_members)
        if not members:
            raise ContractViolationError(f"{_CONTRACT_NAME} group members must not be empty")
        blank_members = [value for value in members if not value]
        if blank_members:
            raise ContractViolationError(f"{_CONTRACT_NAME} group contains blank members")
        outside = sorted(members.difference(universe))
        if outside:
            raise ContractViolationError(f"{_CONTRACT_NAME} group members are outside universe: {outside[:5]}")
        validated.append(CategoricalEnrichmentGroup(labels=labels, members=members))
    if not validated:
        raise ContractViolationError(f"{_CONTRACT_NAME} requires at least one group")
    return validated


def categorical_enrichment_rows(
    *,
    universe_ids: Iterable[str],
    features_by_subject: Mapping[str, Iterable[str]],
    feature_labels: Mapping[str, str],
    groups: Iterable[CategoricalEnrichmentGroup],
    config: CategoricalEnrichmentConfig,
    common_features: Iterable[str],
) -> list[dict[str, object]]:
    """Test categorical-feature enrichment within each named group.

    The primitive is domain-neutral: callers decide whether features are
    regulators, labels, motifs, ontology terms, or other categorical sidecars.
    """

    _validate_config(config)
    subject_ids = _unique_ids(universe_ids, field_name="universe_ids")
    universe = set(subject_ids)
    if not universe:
        raise ContractViolationError(f"{_CONTRACT_NAME} universe_ids must not be empty")
    members_by_feature, labels = _feature_members(
        universe=universe,
        features_by_subject=features_by_subject,
        feature_labels=feature_labels,
    )
    validated_groups = _validated_groups(groups=groups, universe=universe)
    common = {
        str(value).strip().casefold()
        for value in _require_sequence(common_features, field_name="common_features")
        if str(value).strip()
    }
    feature_keys = sorted(labels, key=lambda key: labels[key].casefold())
    total = len(subject_ids)
    output_rows: list[dict[str, object]] = []
    p_values: list[float] = []
    for group in validated_groups:
        group_members = set(group.members)
        group_size = len(group_members)
        for feature in feature_keys:
            feature_members = members_by_feature[feature]
            feature_total = len(feature_members)
            feature_group_hits = len(feature_members.intersection(group_members))
            a = feature_group_hits
            b = feature_total - feature_group_hits
            c = group_size - feature_group_hits
            d = total - a - b - c
            group_fraction = float("nan") if group_size == 0 else a / float(group_size)
            background_fraction = feature_total / float(total)
            enrichment_ratio = (
                float("nan") if group_size == 0 or background_fraction == 0.0 else group_fraction / background_fraction
            )
            p_value = hypergeometric_survival(
                observed=a,
                population=total,
                successes=feature_total,
                draws=group_size,
            )
            p_values.append(p_value)
            passes_min_feature_support = feature_total >= config.min_feature_support
            passes_min_group_hits = feature_group_hits >= config.min_group_hits
            is_common_feature = feature.casefold() in common or labels[feature].casefold() in common
            notes = []
            if not passes_min_feature_support:
                notes.append("below_min_feature_support")
            if not passes_min_group_hits:
                notes.append("below_min_group_hits")
            if is_common_feature:
                notes.append("common_feature")
            output_rows.append(
                {
                    **dict(group.labels),
                    "feature": labels[feature],
                    "n_total": total,
                    "n_group": group_size,
                    "n_feature_total": feature_total,
                    "n_feature_group": feature_group_hits,
                    "group_fraction": float(group_fraction),
                    "background_fraction": float(background_fraction),
                    "enrichment_ratio": float(enrichment_ratio),
                    "odds_ratio": odds_ratio(a, b, c, d),
                    "p_value": float(p_value),
                    "q_value": float("nan"),
                    "p_value_method": config.p_value_method,
                    "fdr_method": config.fdr_method,
                    "passes_min_feature_support": passes_min_feature_support,
                    "passes_min_group_hits": passes_min_group_hits,
                    "is_common_feature": is_common_feature,
                    "notes": ";".join(notes),
                }
            )
    q_values = benjamini_hochberg(p_values)
    for row, q_value in zip(output_rows, q_values, strict=True):
        row["q_value"] = q_value
    return output_rows
