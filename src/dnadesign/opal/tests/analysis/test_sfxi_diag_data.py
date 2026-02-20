"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_sfxi_diag_data.py

Tests for SFXI diagnostic plot helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import polars as pl
import pytest

from dnadesign.opal.src.core.utils import OpalError
from dnadesign.opal.src.plots.sfxi_diag_data import (
    parse_delta_from_runs,
    parse_exponents_from_runs,
)


def test_parse_exponents_from_runs_reads_objective_params() -> None:
    runs_df = pl.DataFrame(
        {
            "objective__params": [
                {
                    "logic_exponent_beta": 1.5,
                    "intensity_exponent_gamma": 0.7,
                }
            ]
        }
    )

    beta, gamma = parse_exponents_from_runs(runs_df)

    assert beta == pytest.approx(1.5)
    assert gamma == pytest.approx(0.7)


def test_parse_exponents_from_runs_requires_named_keys() -> None:
    runs_df = pl.DataFrame({"objective__params": [{"beta": 1.0, "gamma": 1.0}]})

    with pytest.raises(OpalError):
        parse_exponents_from_runs(runs_df)


def test_parse_delta_from_runs_reads_objective_params() -> None:
    runs_df = pl.DataFrame({"objective__params": [{"intensity_log2_offset_delta": 0.25}]})

    assert parse_delta_from_runs(runs_df) == pytest.approx(0.25)


def test_parse_delta_from_runs_requires_key() -> None:
    runs_df = pl.DataFrame({"objective__params": [{}]})

    with pytest.raises(OpalError):
        parse_delta_from_runs(runs_df)
