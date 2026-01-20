"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_run_id_round_resolution.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from dnadesign.opal.src.analysis.facade import read_predictions
from dnadesign.opal.src.core.utils import OpalError
from dnadesign.opal.src.storage.ledger import LedgerError, LedgerReader
from dnadesign.opal.src.storage.workspace import CampaignWorkspace


def _write_pred_parts(pred_dir: Path) -> None:
    pred_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "run_id": ["run-a", "run-b"],
            "as_of_round": [1, 2],
            "id": ["a1", "b1"],
            "pred__y_obj_scalar": [0.1, 0.2],
        }
    ).write_parquet(pred_dir / "part-0.parquet")


def test_facade_read_predictions_run_id_resolves_round(tmp_path: Path) -> None:
    pred_dir = tmp_path / "ledger.predictions"
    _write_pred_parts(pred_dir)
    runs_df = pl.DataFrame({"run_id": ["run-a", "run-b"], "as_of_round": [1, 2]})

    df = read_predictions(
        pred_dir,
        columns=["run_id", "as_of_round", "id"],
        run_id="run-a",
        runs_df=runs_df,
        round_selector=None,
    )
    assert df.height == 1
    assert df["run_id"][0] == "run-a"
    assert int(df["as_of_round"][0]) == 1


def test_facade_read_predictions_run_id_round_mismatch_raises(tmp_path: Path) -> None:
    pred_dir = tmp_path / "ledger.predictions"
    _write_pred_parts(pred_dir)
    runs_df = pl.DataFrame({"run_id": ["run-a", "run-b"], "as_of_round": [1, 2]})

    with pytest.raises(OpalError):
        read_predictions(
            pred_dir,
            columns=["run_id", "as_of_round", "id"],
            run_id="run-a",
            runs_df=runs_df,
            round_selector=[2],
        )


def test_facade_read_predictions_requires_runs_df(tmp_path: Path) -> None:
    pred_dir = tmp_path / "ledger.predictions"
    _write_pred_parts(pred_dir)

    with pytest.raises(OpalError):
        read_predictions(
            pred_dir,
            columns=["run_id", "as_of_round", "id"],
            run_id="run-a",
            runs_df=None,
            round_selector="all",
        )


def test_ledger_reader_run_id_resolves_round(tmp_path: Path) -> None:
    ws = CampaignWorkspace(config_path=tmp_path / "campaign.yaml", workdir=tmp_path)
    _write_pred_parts(ws.ledger_predictions_dir)
    pd.DataFrame({"run_id": ["run-a", "run-b"], "as_of_round": [1, 2]}).to_parquet(ws.ledger_runs_path)

    reader = LedgerReader(ws)
    df = reader.read_predictions(columns=["run_id", "as_of_round"], run_id="run-a", round_selector=None)
    assert df.shape[0] == 1
    assert df.iloc[0]["run_id"] == "run-a"
    assert int(df.iloc[0]["as_of_round"]) == 1


def test_ledger_reader_run_id_round_mismatch_raises(tmp_path: Path) -> None:
    ws = CampaignWorkspace(config_path=tmp_path / "campaign.yaml", workdir=tmp_path)
    _write_pred_parts(ws.ledger_predictions_dir)
    pd.DataFrame({"run_id": ["run-a", "run-b"], "as_of_round": [1, 2]}).to_parquet(ws.ledger_runs_path)

    reader = LedgerReader(ws)
    with pytest.raises(LedgerError):
        reader.read_predictions(columns=["run_id"], run_id="run-a", round_selector=2)
