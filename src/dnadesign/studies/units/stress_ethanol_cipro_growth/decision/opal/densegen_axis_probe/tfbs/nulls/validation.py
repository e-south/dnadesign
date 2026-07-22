"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/densegen_axis_probe/tfbs/nulls/validation.py

Algebra and distribution validators for TFBS permutation nulls.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from .contracts import TFBS_SLOT_COUNT_MATCH_COLUMNS


def _recompute_slot_event_columns(frame: pd.DataFrame) -> None:
    slot0 = frame["slot0_family"].astype(str)
    slot1 = frame["slot1_family"].astype(str)
    slot2 = frame["slot2_family"].astype(str)
    frame["lexA_in_slot0"] = slot0.eq("LexA").astype(int)
    frame["lexA_in_slot1"] = slot1.eq("LexA").astype(int)
    frame["lexA_in_slot2"] = slot2.eq("LexA").astype(int)
    frame["baeR_in_slot1"] = slot1.eq("BaeR").astype(int)
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
    expected_baer_middle = frame["slot1_family"].astype(str).eq("BaeR").astype(int)
    if not expected_baer_middle.reset_index(drop=True).equals(frame["baeR_in_slot1"].reset_index(drop=True)):
        raise ValueError("slot labels are inconsistent with baeR_in_slot1")
