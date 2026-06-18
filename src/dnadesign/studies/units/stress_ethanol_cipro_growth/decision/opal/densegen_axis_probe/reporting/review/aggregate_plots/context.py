"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/densegen_axis_probe/reporting/review/aggregate_plots/context.py

Data context for DenseGen axis probe aggregate plot plugins.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Mapping, Sequence

import pandas as pd

from .source_frames import feature_stability_rows, vector_reference_distance_rows


@dataclass(frozen=True)
class ProbeAggregatePlotContext:
    """Normalized inputs shared by registered aggregate plot renderers."""

    runs_frame: pd.DataFrame
    metrics_payload: Mapping[str, Any]
    configured_plots: Sequence[Mapping[str, Any]]

    @classmethod
    def from_payload(
        cls,
        *,
        metrics_payload: Mapping[str, Any],
        configured_plots: Sequence[Mapping[str, Any]],
    ) -> "ProbeAggregatePlotContext":
        runs = metrics_payload.get("runs") or []
        run_rows = [row for row in runs if isinstance(row, Mapping)]
        if not run_rows:
            raise ValueError("probe aggregate plots require at least one run metric row")
        return cls(
            runs_frame=pd.DataFrame(run_rows),
            metrics_payload=metrics_payload,
            configured_plots=tuple(row for row in configured_plots if isinstance(row, Mapping)),
        )

    @cached_property
    def round_frame(self) -> pd.DataFrame:
        rows = [row for row in self.metrics_payload.get("rounds") or [] if isinstance(row, Mapping)]
        return pd.DataFrame(rows)

    @cached_property
    def trajectory_qa(self) -> Mapping[str, Any]:
        payload = self.metrics_payload.get("trajectory_qa")
        return payload if isinstance(payload, Mapping) else {}

    @cached_property
    def vector_reference_distance_frame(self) -> pd.DataFrame:
        return pd.DataFrame(vector_reference_distance_rows(self.configured_plots))

    @cached_property
    def feature_stability_frame(self) -> pd.DataFrame:
        return pd.DataFrame(feature_stability_rows(self.configured_plots))
