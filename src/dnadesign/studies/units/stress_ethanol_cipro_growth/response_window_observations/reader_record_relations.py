"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/response_window_observations/reader_record_relations.py

Validate coverage relations among canonical Reader event-window records.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def validate_reader_record_relations(
    *,
    designs: pd.DataFrame,
    draws: pd.DataFrame,
    wells: pd.DataFrame,
    traces: pd.DataFrame,
    sources: Sequence[str],
    reductions: set[str],
    expected_draws: int,
) -> None:
    """Require full source, reduction, reference, and draw coverage."""

    reduction_frames = {
        "designs": designs,
        "descriptive_resampling_draws": draws,
        "wells": wells,
    }
    expected_source_reductions = {(source, reduction) for source in sources for reduction in reductions}
    for label, frame in reduction_frames.items():
        observed = _identity_set(frame, ["experiment_id", "reduction_id"])
        if observed != expected_source_reductions:
            raise ValueError(f"Reader {label} does not cover every projected source and reduction.")
        reference = _identity_set(frame.loc[frame["is_reference"]], ["experiment_id", "reduction_id"])
        if reference != expected_source_reductions:
            raise ValueError(f"Reader {label} reference does not cover every source and reduction.")

    expected_reduction_identities = _identity_set(designs, ["experiment_id", "design_id", "reduction_id"])
    for label, frame in (("descriptive_resampling_draws", draws), ("wells", wells)):
        if _identity_set(frame, ["experiment_id", "design_id", "reduction_id"]) != expected_reduction_identities:
            raise ValueError(f"Reader {label} design coverage disagrees with designs.")
    expected_designs = _identity_set(designs, ["experiment_id", "design_id"])
    if _identity_set(traces, ["experiment_id", "design_id"]) != expected_designs:
        raise ValueError("Reader traces design coverage disagrees with designs.")

    observed_draw_identities: set[tuple[str, ...]] = set()
    for identity, group in draws.groupby(["experiment_id", "design_id", "reduction_id"], sort=False):
        observed_draw_identities.add(tuple(str(value) for value in identity))
        if sorted(group["draw_index"].astype(int).tolist()) != list(range(expected_draws)):
            raise ValueError(f"Reader descriptive-resampling draws for {identity!r} are incomplete.")
    if observed_draw_identities != expected_reduction_identities:
        raise ValueError("Reader descriptive-resampling draw identities disagree with designs.")


def _identity_set(frame: pd.DataFrame, columns: list[str]) -> set[tuple[str, ...]]:
    return set(frame.loc[:, columns].astype(str).drop_duplicates().itertuples(index=False, name=None))


__all__ = ["validate_reader_record_relations"]
