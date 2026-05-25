"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/runtime/test_prediction_progress.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from dnadesign.opal.src.runtime.round.stages.prediction import _predict_batch_total


def test_predict_batch_total_never_underruns_seen_batch_index() -> None:
    assert _predict_batch_total(estimated_batches=123, batch_index=124) == 124
    assert _predict_batch_total(estimated_batches=614, batch_index=17) == 614
