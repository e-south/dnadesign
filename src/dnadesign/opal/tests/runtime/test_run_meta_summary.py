"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/runtime/test_run_meta_summary.py

Regression tests for run meta summary OPAL runtime.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import json

import pandas as pd
import pytest

from dnadesign.opal.src.core.utils import OpalError
from dnadesign.opal.src.reporting.summary import select_run_meta, summarize_run_meta


def test_select_run_meta_latest_round():
    df = pd.DataFrame(
        {
            "run_id": ["r0-a", "r1-b"],
            "as_of_round": [0, 1],
            "model__name": ["m0", "m1"],
            "objective__defs_json": [json.dumps([]), json.dumps([])],
            "selection_views__defs_json": [json.dumps([]), json.dumps([])],
            "training__y_ops": [[], []],
            "stats__n_train": [10, 11],
            "stats__n_scored": [100, 110],
        }
    )
    row = select_run_meta(df)
    summary = summarize_run_meta(row)
    assert summary["run_id"] == "r1-b"
    assert summary["as_of_round"] == 1


def test_select_run_meta_requires_run_id_for_duplicates():
    df = pd.DataFrame(
        {
            "run_id": ["r0-a", "r0-b"],
            "as_of_round": [0, 0],
            "model__name": ["m0", "m0"],
            "objective__defs_json": [json.dumps([]), json.dumps([])],
            "selection_views__defs_json": [json.dumps([]), json.dumps([])],
            "training__y_ops": [[], []],
            "stats__n_train": [10, 10],
            "stats__n_scored": [100, 100],
        }
    )
    with pytest.raises(OpalError):
        _ = select_run_meta(df, round_sel=0)
