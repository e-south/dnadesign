"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/runtime/multistate_behavior_event_verification.py

Verify persisted multistate behavior event-sensitivity evidence.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def verify_event_score_derivations(event: pd.DataFrame, observed: pd.DataFrame) -> None:
    """Verify central values, directional envelopes, and unit ranks."""

    aligned = event.merge(
        observed.loc[:, ["id", "selection_view_id", "behavior_score"]],
        on=["id", "selection_view_id"],
        how="left",
        validate="one_to_one",
    )
    if aligned["behavior_score"].isna().any() or not np.allclose(
        aligned["behavior_score_central"],
        aligned["behavior_score"],
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError("event central scores disagree with observed behavior scores.")
    if not (
        (aligned["behavior_score_worst_envelope"] <= aligned["behavior_score_central"])
        & (aligned["behavior_score_central"] <= aligned["behavior_score_best_envelope"])
    ).all():
        raise ValueError("event score envelopes do not contain their central scores.")
    if not np.allclose(
        aligned["behavior_score_envelope_width"],
        aligned["behavior_score_best_envelope"] - aligned["behavior_score_worst_envelope"],
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError("event score envelope widths do not derive from their bounds.")
    if not (
        aligned["event_unit_rank_min"].eq(
            aligned[["central_unit_rank", "worst_envelope_unit_rank", "best_envelope_unit_rank"]].min(axis=1)
        )
        & aligned["event_unit_rank_max"].eq(
            aligned[["central_unit_rank", "worst_envelope_unit_rank", "best_envelope_unit_rank"]].max(axis=1)
        )
        & aligned["event_unit_rank_span"].eq(aligned["event_unit_rank_max"] - aligned["event_unit_rank_min"])
    ).all():
        raise ValueError("event rank envelopes do not derive from their component ranks.")
    if not (aligned["hard_bottleneck_worst_envelope"] <= aligned["hard_bottleneck_best_envelope"]).all():
        raise ValueError("event hard-bottleneck envelopes are directionally inconsistent.")
    for view_id, rows in aligned.groupby("selection_view_id", sort=False):
        _verify_unit_ranks(rows, view_id=str(view_id))


def _verify_unit_ranks(rows: pd.DataFrame, *, view_id: str) -> None:
    contracts = (
        ("behavior_score_central", "central_unit_rank"),
        ("behavior_score_worst_envelope", "worst_envelope_unit_rank"),
        ("behavior_score_best_envelope", "best_envelope_unit_rank"),
    )
    for score_column, rank_column in contracts:
        ranked = rows.assign(id=rows["id"].astype(str)).sort_values(
            [score_column, "id"],
            ascending=[False, True],
            kind="mergesort",
        )
        expected = pd.Series(np.arange(1, len(ranked) + 1, dtype=int), index=ranked["id"].astype(str))
        observed_ranks = rows["id"].astype(str).map(expected).to_numpy(dtype=int)
        if not np.array_equal(rows[rank_column].to_numpy(dtype=int), observed_ranks):
            raise ValueError(f"event {rank_column} does not derive from {score_column} for {view_id!r}.")


__all__ = ["verify_event_score_derivations"]
