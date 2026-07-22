"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/runtime/run_contracts.py

Shared SFXI-evidence contracts for the response metric metastudy runtime.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.contracts import SfxiEvidenceFrame


def assert_shared_label_sources(frames: tuple[pd.DataFrame, ...]) -> None:
    if not frames:
        raise ValueError("at least one Reader label source is required.")
    columns = [
        "id",
        "design_id",
        "reader_experiment_id",
        "v00",
        "v10",
        "v01",
        "v11",
        "y00_star",
        "y10_star",
        "y01_star",
        "y11_star",
    ]
    baseline = frames[0].loc[:, columns].reset_index(drop=True)
    for frame in frames[1:]:
        if not baseline.equals(frame.loc[:, columns].reset_index(drop=True)):
            raise ValueError("Reader label sources are not identical across SFXI evidence inputs.")


def predictor_parity(sfxi_evidence: tuple[SfxiEvidenceFrame, ...]) -> dict[str, object]:
    first = sfxi_evidence[0]
    max_abs = max(float(np.max(np.abs(frame.y_hat - first.y_hat))) for frame in sfxi_evidence[1:])
    if max_abs > 1.0e-12:
        raise RuntimeError(f"SFXI evidence predictor surfaces diverge; max_abs_error={max_abs}.")
    if any(dict(frame.model_params) != dict(first.model_params) for frame in sfxi_evidence[1:]):
        raise RuntimeError("SFXI evidence random-forest parameters diverge.")
    return {
        "shared_predictor": True,
        "max_abs_prediction_difference": max_abs,
        "interpretation": "The SFXI inputs are target views over one shared vec8 predictor, not independent models.",
    }
