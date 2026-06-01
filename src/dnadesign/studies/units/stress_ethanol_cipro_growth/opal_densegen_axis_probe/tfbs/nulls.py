"""Matched null construction for the DenseGen TFBS learnability probe v1."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .oracle import validate_tfbs_label_algebra
from .schema import (
    TFBS_LEARNABILITY_ACTIVE_LABEL_NAMES,
    TFBS_LEARNABILITY_FAMILY_CONTENT_NULL_VERSION,
    TFBS_LEARNABILITY_LABEL_RECIPE_HASH,
    TFBS_LEARNABILITY_NULL_VIABILITY_STATUSES,
    TFBS_LEARNABILITY_ORACLE_VERSION,
    TFBS_LEARNABILITY_SLOT_GEOMETRY_NULL_VERSION,
)

TFBS_COUNT_COLUMNS = ("lexA_count", "cpxR_count", "baeR_count", "cpxR_or_baeR_count")
TFBS_PRESENCE_COLUMNS = ("lexA_present", "cpxR_present", "baeR_present", "cpxR_or_baeR_present")
TFBS_COUNT_FRACTION_COLUMNS = (
    "lexA_count_fraction",
    "cpxR_count_fraction",
    "baeR_count_fraction",
    "cpxR_or_baeR_count_fraction",
)
TFBS_SLOT_EVENT_COLUMNS = (
    "lexA_in_slot0",
    "lexA_in_slot1",
    "lexA_in_slot2",
    "cpxR_or_baeR_in_slot0",
    "cpxR_or_baeR_in_slot1",
    "cpxR_or_baeR_in_slot2",
)
TFBS_SLOT_FAMILY_COLUMNS = ("slot0_family", "slot1_family", "slot2_family")
TFBS_PASSIVE_STRATUM_COLUMNS = ("sigma35_variant", "spacer_length")
TFBS_CONTENT_BLOCK_COLUMNS = (
    *TFBS_COUNT_COLUMNS,
    *TFBS_PRESENCE_COLUMNS,
    *TFBS_COUNT_FRACTION_COLUMNS,
    *TFBS_SLOT_EVENT_COLUMNS,
    *TFBS_SLOT_FAMILY_COLUMNS,
)
TFBS_SLOT_COUNT_MATCH_COLUMNS = ("lexA_count", "cpxR_count", "baeR_count")


@dataclass(frozen=True)
class TfbsNullConfig:
    """Viability thresholds for matched TFBS permutation nulls."""

    tiny_stratum_threshold: int = 3
    fail_if_fraction_rows_in_singleton_strata_gt: float = 0.01
    fail_if_fraction_rows_in_tiny_strata_gt: float = 0.05
    fail_on_weak_exchangeability: bool = True
    warn_if_unchanged_label_fraction_ge: float = 0.50
    fail_if_unchanged_label_fraction_ge: float = 0.75


@dataclass(frozen=True)
class TfbsNullBuild:
    """Null label table plus the manifest-ready viability report."""

    labels: pd.DataFrame
    null_viability_report: dict[str, Any]


def build_tfbs_family_content_matched_null(
    labels: pd.DataFrame,
    *,
    seed: int,
    label_name: str = "tf_family_content_block",
    stratum_candidates: Sequence[Sequence[str]] = (
        ("sigma35_variant", "spacer_length"),
        ("sigma35_variant",),
        (),
    ),
    config: TfbsNullConfig | None = None,
) -> TfbsNullBuild:
    """Permute the v1 TFBS content-label block within matched sigma-core strata."""

    cfg = config or TfbsNullConfig()
    frame = labels.reset_index(drop=True).copy()
    _require_columns(frame, ("id", "quality_flag", *TFBS_CONTENT_BLOCK_COLUMNS, *TFBS_PASSIVE_STRATUM_COLUMNS))
    selected = _select_viable_stratum(frame, stratum_candidates=stratum_candidates, config=cfg)
    donor_positions = _permuted_donor_positions(frame, selected.stratum_columns, seed=seed)
    out = frame.copy()
    donor = frame.iloc[donor_positions].reset_index(drop=True)
    for column in TFBS_CONTENT_BLOCK_COLUMNS:
        out[column] = donor[column].to_numpy()
    _validate_label_distribution(frame, out, columns=TFBS_ACTIVE_NUMERIC_COLUMNS)
    validate_tfbs_label_algebra(out)
    _validate_slot_label_consistency(out)
    report = _null_viability_report(
        before=frame,
        after=out,
        null_version=TFBS_LEARNABILITY_FAMILY_CONTENT_NULL_VERSION,
        seed=seed,
        label_name=label_name,
        selected=selected,
        config=cfg,
        compare_columns=TFBS_CONTENT_BLOCK_COLUMNS,
        label_joint_columns=TFBS_ACTIVE_NUMERIC_COLUMNS,
        null_control_role="matched_label_permutation_negative_control",
        preserved_signal="sigma-core stratum and label marginal distributions",
        disrupted_signal="row association between sequence identity and TFBS content labels",
        negative_control_claim_status="VALID_AS_NEGATIVE_CONTROL",
        warnings=_content_warnings(selected),
    )
    return TfbsNullBuild(
        labels=_with_null_metadata(
            out,
            null_version=TFBS_LEARNABILITY_FAMILY_CONTENT_NULL_VERSION,
            null_control_role="matched_label_permutation_negative_control",
            negative_control_claim_status="VALID_AS_NEGATIVE_CONTROL",
            seed=seed,
            positive_labels=frame,
            selected=selected,
        ),
        null_viability_report=report,
    )


def build_tfbs_slot_geometry_count_matched_null(
    labels: pd.DataFrame,
    *,
    label_name: str,
    seed: int,
    stratum_candidates: Sequence[Sequence[str]] = (
        ("sigma35_variant", "spacer_length", "lexA_count", "cpxR_count", "baeR_count"),
        ("sigma35_variant", "lexA_count", "cpxR_count", "baeR_count"),
        ("lexA_count", "cpxR_count", "baeR_count"),
    ),
    config: TfbsNullConfig | None = None,
) -> TfbsNullBuild:
    """Permute slot-family geometry while preserving row-level TF family counts."""

    if label_name not in TFBS_SLOT_EVENT_COLUMNS:
        raise ValueError(f"slot-geometry null label_name must be a v1 slot label, got {label_name!r}")
    cfg = config or TfbsNullConfig()
    frame = labels.reset_index(drop=True).copy()
    _require_columns(
        frame,
        (
            "id",
            "quality_flag",
            *TFBS_SLOT_FAMILY_COLUMNS,
            *TFBS_SLOT_EVENT_COLUMNS,
            *TFBS_SLOT_COUNT_MATCH_COLUMNS,
            *TFBS_PASSIVE_STRATUM_COLUMNS,
        ),
    )
    selected = _select_viable_stratum(frame, stratum_candidates=stratum_candidates, config=cfg)
    donor_positions = _permuted_donor_positions(frame, selected.stratum_columns, seed=seed)
    out = frame.copy()
    donor = frame.iloc[donor_positions].reset_index(drop=True)
    for column in TFBS_SLOT_FAMILY_COLUMNS:
        out[column] = donor[column].to_numpy()
    _recompute_slot_event_columns(out)
    _validate_count_matching(frame, out)
    _validate_label_distribution(frame, out, columns=TFBS_SLOT_EVENT_COLUMNS)
    validate_tfbs_label_algebra(out)
    _validate_slot_label_consistency(out)
    report = _null_viability_report(
        before=frame,
        after=out,
        null_version=TFBS_LEARNABILITY_SLOT_GEOMETRY_NULL_VERSION,
        seed=seed,
        label_name=label_name,
        selected=selected,
        config=cfg,
        compare_columns=(label_name,),
        label_joint_columns=(*TFBS_SLOT_FAMILY_COLUMNS, *TFBS_SLOT_EVENT_COLUMNS),
        null_control_role="count_preserving_slot_confound_control",
        preserved_signal="row-level TF family counts",
        disrupted_signal="slot-family assignment conditional on preserved counts",
        negative_control_claim_status="CONFOUND_CONTROL_ONLY",
        warnings=_slot_warnings(selected),
    )
    return TfbsNullBuild(
        labels=_with_null_metadata(
            out,
            null_version=TFBS_LEARNABILITY_SLOT_GEOMETRY_NULL_VERSION,
            null_control_role="count_preserving_slot_confound_control",
            negative_control_claim_status="CONFOUND_CONTROL_ONLY",
            seed=seed,
            positive_labels=frame,
            selected=selected,
        ),
        null_viability_report=report,
    )


TFBS_ACTIVE_NUMERIC_COLUMNS = tuple(TFBS_LEARNABILITY_ACTIVE_LABEL_NAMES)


@dataclass(frozen=True)
class _SelectedStratum:
    stratum_columns: tuple[str, ...]
    stratum_count: int
    min_rows_per_stratum: int
    median_rows_per_stratum: float
    max_rows_per_stratum: int
    fraction_rows_in_singleton_strata: float
    fraction_rows_in_tiny_strata: float
    viability_status: str
    coarsening_steps_applied: tuple[str, ...]


def _select_viable_stratum(
    frame: pd.DataFrame,
    *,
    stratum_candidates: Sequence[Sequence[str]],
    config: TfbsNullConfig,
) -> _SelectedStratum:
    if not stratum_candidates:
        raise ValueError("at least one null stratum candidate is required")
    for index, candidate in enumerate(stratum_candidates):
        columns = tuple(str(column) for column in candidate)
        _require_columns(frame, columns)
        stats = _stratum_stats(
            frame,
            columns,
            config=config,
            status="PASS" if index == 0 else "PASS_WITH_COARSENING",
            coarsening_steps=tuple(_coarsening_steps(stratum_candidates[: index + 1])),
        )
        if _stratum_passes(stats, config):
            return stats
    failed = _stratum_stats(
        frame,
        tuple(str(column) for column in stratum_candidates[-1]),
        config=config,
        status="FAIL_WEAK_EXCHANGEABILITY",
        coarsening_steps=tuple(_coarsening_steps(stratum_candidates)),
    )
    if config.fail_on_weak_exchangeability:
        raise ValueError(
            "matched null exchangeability is too weak: "
            f"singleton_fraction={failed.fraction_rows_in_singleton_strata:.6g}, "
            f"tiny_fraction={failed.fraction_rows_in_tiny_strata:.6g}, "
            f"stratum_key={_stratum_key_label(failed.stratum_columns)}"
        )
    return failed


def _stratum_stats(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    config: TfbsNullConfig,
    status: str,
    coarsening_steps: tuple[str, ...],
) -> _SelectedStratum:
    if status not in TFBS_LEARNABILITY_NULL_VIABILITY_STATUSES:
        raise ValueError(f"unsupported TFBS null viability status: {status}")
    group_sizes = np.array([len(indices) for indices in _group_indices(frame, columns)], dtype=float)
    if len(group_sizes) == 0:
        raise ValueError("cannot build null strata for an empty label table")
    row_count = float(len(frame))
    singleton_rows = float(group_sizes[group_sizes == 1].sum())
    tiny_rows = float(group_sizes[group_sizes < config.tiny_stratum_threshold].sum())
    return _SelectedStratum(
        stratum_columns=columns,
        stratum_count=int(len(group_sizes)),
        min_rows_per_stratum=int(group_sizes.min()),
        median_rows_per_stratum=float(np.median(group_sizes)),
        max_rows_per_stratum=int(group_sizes.max()),
        fraction_rows_in_singleton_strata=singleton_rows / row_count,
        fraction_rows_in_tiny_strata=tiny_rows / row_count,
        viability_status=status,
        coarsening_steps_applied=coarsening_steps,
    )


def _stratum_passes(selected: _SelectedStratum, config: TfbsNullConfig) -> bool:
    return (
        selected.fraction_rows_in_singleton_strata <= config.fail_if_fraction_rows_in_singleton_strata_gt
        and selected.fraction_rows_in_tiny_strata <= config.fail_if_fraction_rows_in_tiny_strata_gt
    )


def _permuted_donor_positions(frame: pd.DataFrame, stratum_columns: tuple[str, ...], *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    donor_positions = np.arange(len(frame), dtype=int)
    for positions in _group_indices(frame, stratum_columns):
        if len(positions) <= 1:
            continue
        permuted = rng.permutation(positions)
        if np.array_equal(permuted, positions):
            permuted = np.roll(positions, -1)
        donor_positions[positions] = permuted
    return donor_positions


def _group_indices(frame: pd.DataFrame, columns: tuple[str, ...]) -> list[np.ndarray]:
    if not columns:
        return [np.arange(len(frame), dtype=int)]
    key_frame = frame.loc[:, list(columns)].astype(str)
    keys = key_frame.agg("\x1f".join, axis=1)
    groups: list[np.ndarray] = []
    for key in sorted(keys.unique()):
        groups.append(np.flatnonzero(keys.to_numpy() == key))
    return groups


def _recompute_slot_event_columns(frame: pd.DataFrame) -> None:
    slot0 = frame["slot0_family"].astype(str)
    slot1 = frame["slot1_family"].astype(str)
    slot2 = frame["slot2_family"].astype(str)
    frame["lexA_in_slot0"] = slot0.eq("LexA").astype(int)
    frame["lexA_in_slot1"] = slot1.eq("LexA").astype(int)
    frame["lexA_in_slot2"] = slot2.eq("LexA").astype(int)
    frame["cpxR_or_baeR_in_slot0"] = slot0.isin({"CpxR", "BaeR"}).astype(int)
    frame["cpxR_or_baeR_in_slot1"] = slot1.isin({"CpxR", "BaeR"}).astype(int)
    frame["cpxR_or_baeR_in_slot2"] = slot2.isin({"CpxR", "BaeR"}).astype(int)


def _validate_count_matching(before: pd.DataFrame, after: pd.DataFrame) -> None:
    for column in TFBS_SLOT_COUNT_MATCH_COLUMNS:
        if not before[column].reset_index(drop=True).equals(after[column].reset_index(drop=True)):
            raise ValueError(f"slot-geometry null changed row-level count column: {column}")


def _validate_label_distribution(before: pd.DataFrame, after: pd.DataFrame, *, columns: Sequence[str]) -> None:
    changed = []
    for column in columns:
        left = before[column].value_counts(dropna=False).sort_index()
        right = after[column].value_counts(dropna=False).sort_index()
        if not left.equals(right):
            changed.append(column)
    if changed:
        raise ValueError(f"null permutation changed label marginal distribution(s): {changed}")


def _validate_slot_label_consistency(frame: pd.DataFrame) -> None:
    lex_a_slot_sum = frame[["lexA_in_slot0", "lexA_in_slot1", "lexA_in_slot2"]].sum(axis=1)
    if not lex_a_slot_sum.reset_index(drop=True).equals(frame["lexA_count"].reset_index(drop=True)):
        raise ValueError("slot labels are inconsistent with lexA_count")
    cpxr_baer_slot_sum = frame[["cpxR_or_baeR_in_slot0", "cpxR_or_baeR_in_slot1", "cpxR_or_baeR_in_slot2"]].sum(axis=1)
    if not cpxr_baer_slot_sum.reset_index(drop=True).equals(frame["cpxR_or_baeR_count"].reset_index(drop=True)):
        raise ValueError("slot labels are inconsistent with cpxR_or_baeR_count")


def _null_viability_report(
    *,
    before: pd.DataFrame,
    after: pd.DataFrame,
    null_version: str,
    seed: int,
    label_name: str,
    selected: _SelectedStratum,
    config: TfbsNullConfig,
    compare_columns: Sequence[str],
    label_joint_columns: Sequence[str],
    null_control_role: str,
    preserved_signal: str,
    disrupted_signal: str,
    negative_control_claim_status: str,
    warnings: Sequence[str],
) -> dict[str, Any]:
    unchanged_fraction = _unchanged_fraction(before, after, columns=compare_columns)
    leakage = _label_leakage_assessment(unchanged_fraction, config=config)
    report = {
        "schema_version": "stress_ethanol_cipro_growth.densegen_tfbs_null_viability.v1",
        "oracle_version": TFBS_LEARNABILITY_ORACLE_VERSION,
        "null_version": null_version,
        "null_control_role": null_control_role,
        "preserved_signal": preserved_signal,
        "disrupted_signal": disrupted_signal,
        "negative_control_claim_status": negative_control_claim_status,
        "seed": int(seed),
        "row_count": int(len(before)),
        "label_name": label_name,
        "stratum_key": list(selected.stratum_columns),
        "stratum_count": selected.stratum_count,
        "min_rows_per_stratum": selected.min_rows_per_stratum,
        "median_rows_per_stratum": selected.median_rows_per_stratum,
        "max_rows_per_stratum": selected.max_rows_per_stratum,
        "fraction_rows_in_singleton_strata": selected.fraction_rows_in_singleton_strata,
        "fraction_rows_in_tiny_strata": selected.fraction_rows_in_tiny_strata,
        "configured_tiny_stratum_threshold": config.tiny_stratum_threshold,
        "unchanged_label_fraction_after_permutation": unchanged_fraction,
        "label_leakage_assessment": leakage,
        "label_marginal_before": _label_marginal(before, label_name),
        "label_marginal_after": _label_marginal(after, label_name),
        "label_joint_summary_before": _joint_summary(before, label_joint_columns),
        "label_joint_summary_after": _joint_summary(after, label_joint_columns),
        "permutation_entropy": _permutation_entropy(before, selected.stratum_columns),
        "estimated_effective_permutation_count": _effective_permutation_count(before, selected.stratum_columns),
        "coarsening_steps_applied": list(selected.coarsening_steps_applied),
        "viability_status": selected.viability_status,
        "warnings": [*warnings, *_unchanged_warnings(leakage)],
    }
    if report["label_marginal_before"] != report["label_marginal_after"]:
        report["viability_status"] = "FAIL_LABEL_DISTRIBUTION_CHANGED"
    if report["viability_status"] not in TFBS_LEARNABILITY_NULL_VIABILITY_STATUSES:
        raise ValueError(f"unsupported TFBS null viability status: {report['viability_status']}")
    return report


def _with_null_metadata(
    labels: pd.DataFrame,
    *,
    null_version: str,
    null_control_role: str,
    negative_control_claim_status: str,
    seed: int,
    positive_labels: pd.DataFrame,
    selected: _SelectedStratum,
) -> pd.DataFrame:
    out = labels.copy()
    out["null_version"] = null_version
    out["null_control_role"] = null_control_role
    out["negative_control_claim_status"] = negative_control_claim_status
    out["null_seed"] = int(seed)
    out["positive_oracle_version"] = TFBS_LEARNABILITY_ORACLE_VERSION
    out["positive_label_table_hash"] = _dataframe_hash(positive_labels)
    out["null_recipe_hash"] = _null_recipe_hash(null_version=null_version, seed=seed, selected=selected)
    out["stratum_key"] = _stratum_key_label(selected.stratum_columns)
    out["coarsening_steps_applied"] = json.dumps(list(selected.coarsening_steps_applied), separators=(",", ":"))
    out["viability_status"] = selected.viability_status
    return out


def _label_marginal(frame: pd.DataFrame, label_name: str) -> dict[str, int] | None:
    if label_name not in frame.columns:
        return None
    counts = frame[label_name].value_counts(dropna=False).sort_index().to_dict()
    return {str(key): int(value) for key, value in counts.items()}


def _joint_summary(frame: pd.DataFrame, columns: Sequence[str]) -> dict[str, Any]:
    selected_columns = [column for column in columns if column in frame.columns]
    if not selected_columns:
        return {"columns": [], "unique_patterns": 0, "top_patterns": []}
    counts = frame.loc[:, selected_columns].astype(str).value_counts(dropna=False).sort_values(ascending=False).head(20)
    top_patterns = [
        {"pattern": list(key if isinstance(key, tuple) else (key,)), "count": int(value)}
        for key, value in counts.items()
    ]
    return {
        "columns": selected_columns,
        "unique_patterns": int(frame.loc[:, selected_columns].astype(str).drop_duplicates().shape[0]),
        "top_patterns": top_patterns,
        "summary_hash": _summary_hash(frame, selected_columns),
    }


def _unchanged_fraction(before: pd.DataFrame, after: pd.DataFrame, *, columns: Sequence[str]) -> float:
    selected_columns = [column for column in columns if column in before.columns and column in after.columns]
    if not selected_columns:
        return 0.0
    left = before.loc[:, selected_columns].astype(str).agg("\x1f".join, axis=1)
    right = after.loc[:, selected_columns].astype(str).agg("\x1f".join, axis=1)
    return float((left == right).mean())


def _permutation_entropy(frame: pd.DataFrame, stratum_columns: tuple[str, ...]) -> float:
    return float(sum(math.lgamma(len(indices) + 1) for indices in _group_indices(frame, stratum_columns)))


def _effective_permutation_count(frame: pd.DataFrame, stratum_columns: tuple[str, ...]) -> int | str:
    log_count = _permutation_entropy(frame, stratum_columns)
    if log_count > math.log(10) * 100:
        return f">1e{int(log_count / math.log(10))}"
    return int(round(math.exp(log_count)))


def _content_warnings(selected: _SelectedStratum) -> list[str]:
    if selected.viability_status == "PASS_WITH_COARSENING":
        return ["family-content null required declared stratum coarsening"]
    return []


def _slot_warnings(selected: _SelectedStratum) -> list[str]:
    warnings = ["slot-geometry null preserves row-level LexA/CpxR/BaeR counts before permuting slot families"]
    if selected.viability_status == "PASS_WITH_COARSENING":
        warnings.append("slot-geometry null required declared stratum coarsening")
    return warnings


def _label_leakage_assessment(unchanged_fraction: float, *, config: TfbsNullConfig) -> dict[str, Any]:
    if unchanged_fraction >= config.fail_if_unchanged_label_fraction_ge:
        status = "FAIL_HIGH_LABEL_RETENTION"
    elif unchanged_fraction >= config.warn_if_unchanged_label_fraction_ge:
        status = "WARN_HIGH_LABEL_RETENTION"
    else:
        status = "PASS"
    return {
        "status": status,
        "unchanged_label_fraction": unchanged_fraction,
        "warn_if_unchanged_label_fraction_ge": float(config.warn_if_unchanged_label_fraction_ge),
        "fail_if_unchanged_label_fraction_ge": float(config.fail_if_unchanged_label_fraction_ge),
    }


def _unchanged_warnings(leakage: Mapping[str, Any]) -> list[str]:
    status = str(leakage.get("status") or "")
    unchanged_fraction = float(leakage.get("unchanged_label_fraction") or 0.0)
    if status == "FAIL_HIGH_LABEL_RETENTION":
        return [
            "null/control retained too much target-label identity to support a primary negative-control claim "
            f"(unchanged_fraction={unchanged_fraction:.3f})"
        ]
    if status == "WARN_HIGH_LABEL_RETENTION":
        return [
            "null/control retained substantial target-label identity; "
            "interpret as a conservative or confounded control "
            f"(unchanged_fraction={unchanged_fraction:.3f})"
        ]
    return []


def _coarsening_steps(candidates: Sequence[Sequence[str]]) -> list[str]:
    steps = []
    for before, after in zip(candidates, candidates[1:], strict=False):
        steps.append(f"{_stratum_key_label(tuple(before))} -> {_stratum_key_label(tuple(after))}")
    return steps


def _stratum_key_label(columns: Sequence[str]) -> str:
    return "+".join(columns) if columns else "global"


def _null_recipe_hash(*, null_version: str, seed: int, selected: _SelectedStratum) -> str:
    payload = {
        "positive_recipe_hash": TFBS_LEARNABILITY_LABEL_RECIPE_HASH,
        "null_version": null_version,
        "seed": int(seed),
        "stratum_key": list(selected.stratum_columns),
        "coarsening_steps_applied": list(selected.coarsening_steps_applied),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _dataframe_hash(frame: pd.DataFrame) -> str:
    payload = pd.util.hash_pandas_object(frame.astype(str), index=True).to_numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _summary_hash(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    payload = frame.loc[:, list(columns)].astype(str).sort_values(list(columns)).to_json(orient="split")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"TFBS null label frame missing required column(s): {missing}")
