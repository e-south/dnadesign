"""Exchangeability strata and deterministic donor permutation for TFBS nulls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from ..schema import TFBS_LEARNABILITY_NULL_VIABILITY_STATUSES
from .contracts import TfbsNullConfig


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


def _coarsening_steps(candidates: Sequence[Sequence[str]]) -> list[str]:
    steps = []
    for before, after in zip(candidates, candidates[1:], strict=False):
        steps.append(f"{_stratum_key_label(tuple(before))} -> {_stratum_key_label(tuple(after))}")
    return steps


def _stratum_key_label(columns: Sequence[str]) -> str:
    return "+".join(columns) if columns else "global"


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"TFBS null label frame missing required column(s): {missing}")
