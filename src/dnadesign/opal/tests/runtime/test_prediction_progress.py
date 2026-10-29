"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/runtime/test_prediction_progress.py

Regression tests for prediction progress OPAL runtime.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import numpy as np
import pytest

from dnadesign.opal.src.core.utils import OpalError
from dnadesign.opal.src.runtime.round.stages.prediction import (
    _align_predictions_to_requested_order,
    _predict_batch_total,
)


def test_predict_batch_total_never_underruns_seen_batch_index() -> None:
    assert _predict_batch_total(estimated_batches=123, batch_index=124) == 124
    assert _predict_batch_total(estimated_batches=614, batch_index=17) == 614


def test_align_predictions_to_requested_order_reorders_yhat_and_uncertainty() -> None:
    y_hat = np.asarray([[2.0, 20.0], [1.0, 10.0], [3.0, 30.0]])
    y_pred_std = np.asarray([0.2, 0.1, 0.3])

    aligned_y_hat, aligned_std, reordered = _align_predictions_to_requested_order(
        y_hat=y_hat,
        y_pred_std=y_pred_std,
        predicted_ids=["b", "a", "c"],
        requested_ids=["a", "b", "c"],
    )

    assert reordered is True
    assert aligned_y_hat.tolist() == [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]
    assert aligned_std.tolist() == [0.1, 0.2, 0.3]


def test_align_predictions_to_requested_order_rejects_id_coverage_mismatch() -> None:
    with pytest.raises(OpalError, match="coverage mismatch"):
        _align_predictions_to_requested_order(
            y_hat=np.asarray([[1.0], [2.0]]),
            y_pred_std=None,
            predicted_ids=["a", "extra"],
            requested_ids=["a", "missing"],
        )


def test_align_predictions_to_requested_order_rejects_prediction_row_mismatch() -> None:
    with pytest.raises(OpalError, match="row count mismatch"):
        _align_predictions_to_requested_order(
            y_hat=np.asarray([[1.0], [2.0]]),
            y_pred_std=None,
            predicted_ids=["a"],
            requested_ids=["a"],
        )
