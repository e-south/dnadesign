"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/runtime/multistate_behavior_reference.py

Fail-closed pDual-10 reference-relative bootstrap identity checks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ReferenceSignalIdentityReceipt:
    """Counts proving that the same resample was used on both sides of b_i."""

    reference_unit_count: int
    bootstrap_row_count: int
    reader_experiment_count: int


def verify_reference_relative_bootstrap_identity(
    designs: pd.DataFrame,
    bootstrap_draws: pd.DataFrame,
    *,
    primary_reduction_id: str,
    state_ids: tuple[str, ...],
) -> ReferenceSignalIdentityReceipt:
    """Require pDual-10 compared with itself to be exactly zero in every draw."""

    signal_columns = tuple(f"b{state_id}" for state_id in state_ids)
    sd_columns = tuple(f"{column}_bootstrap_sd" for column in signal_columns)
    design_required = {"experiment_id", "design_id", "reduction_id", "is_reference", *signal_columns, *sd_columns}
    draw_required = {
        "experiment_id",
        "design_id",
        "reduction_id",
        "is_reference",
        "draw_index",
        *signal_columns,
    }
    if missing := sorted(design_required - set(designs.columns)):
        raise ValueError(f"Reader reference identity designs lack fields: {missing}")
    if missing := sorted(draw_required - set(bootstrap_draws.columns)):
        raise ValueError(f"Reader reference identity draws lack fields: {missing}")
    reference = designs.loc[
        designs["is_reference"].astype(bool) & designs["reduction_id"].astype(str).eq(primary_reduction_id)
    ].copy()
    draws = bootstrap_draws.loc[
        bootstrap_draws["is_reference"].astype(bool)
        & bootstrap_draws["reduction_id"].astype(str).eq(primary_reduction_id)
    ].copy()
    if reference.empty or draws.empty:
        raise ValueError("Reader bundle lacks primary pDual-10 reference evidence.")
    if reference.duplicated(subset=["experiment_id", "design_id"]).any():
        raise ValueError("Reader primary reference units must be unique by experiment and design.")
    if reference.groupby("experiment_id", sort=False)["design_id"].nunique().ne(1).any():
        raise ValueError("Reader primary reduction must contain one reference design per experiment.")
    reference_keys = set(reference[["experiment_id", "design_id"]].astype(str).itertuples(index=False, name=None))
    draw_keys = set(draws[["experiment_id", "design_id"]].astype(str).itertuples(index=False, name=None))
    if draw_keys != reference_keys:
        raise ValueError("Reader reference bootstrap identities disagree with primary reference units.")
    for frame, columns, context in (
        (reference, signal_columns, "central reference-relative signal"),
        (reference, sd_columns, "reference bootstrap SD"),
        (draws, signal_columns, "reference bootstrap draw"),
    ):
        values = frame.loc[:, list(columns)].to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.equal(values, 0.0).all():
            raise ValueError(
                f"Reader {context} must be definitionally zero when pDual-10 is compared with the same resample."
            )
    for key, rows in draws.groupby(["experiment_id", "design_id"], sort=False):
        indexes = pd.to_numeric(rows["draw_index"], errors="coerce").to_numpy(dtype=float)
        if (
            not np.isfinite(indexes).all()
            or not np.equal(indexes, np.floor(indexes)).all()
            or len(np.unique(indexes)) != len(indexes)
        ):
            raise ValueError(f"Reader reference bootstrap draw indexes are invalid for {key!r}.")
    return ReferenceSignalIdentityReceipt(
        reference_unit_count=len(reference),
        bootstrap_row_count=len(draws),
        reader_experiment_count=int(reference["experiment_id"].astype(str).nunique()),
    )


__all__ = ["ReferenceSignalIdentityReceipt", "verify_reference_relative_bootstrap_identity"]
