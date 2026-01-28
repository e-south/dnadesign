"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/analysis/sfxi/ensemble.py

Ensemble prediction protocol for streaming per-estimator outputs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class SupportsEnsemblePredictions(Protocol):
    def iter_ensemble_predictions(
        self,
        X: np.ndarray,
        *,
        batch_size: int,
    ) -> Iterator[tuple[int, int, int, np.ndarray]]:
        """
        Stream per-estimator predictions as (estimator_index, row_start, row_end, y_pred_batch).

        The iterator must yield batches grouped by estimator; row ranges are zero-based slices
        into X with shape (row_end - row_start, D).
        """
