"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_cli_pressure_sfxi_roundtrip.py

Module Author(s): Eric J. South (extended by Codex)
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from dnadesign.opal.src.cli.app import _build
from dnadesign.opal.src.storage.data_access import RecordsStore

from ._cli_helpers import write_campaign_yaml


def _write_records(path: Path) -> None:
    df = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d", "e"],
            "sequence": ["AAA", "BBB", "CCC", "DDD", "EEE"],
            "bio_type": ["dna"] * 5,
            "alphabet": ["dna_4"] * 5,
            "X": [[0.1], [0.2], [0.3], [0.4], [0.5]],
        }
    )
    df.to_parquet(path, index=False)


def _write_sfxi_labels(path: Path, *, seqs: list[str]) -> None:
    n = len(seqs)
    df = pd.DataFrame(
        {
            "sequence": seqs,
            "v00": [0.0] * n,
            "v10": [0.0] * n,
            "v01": [0.0] * n,
            "v11": [1.0] * n,
            "y00_star": [0.1] * n,
            "y10_star": [0.2] * n,
            "y01_star": [0.3] * n,
            "y11_star": [0.4] * n,
            "intensity_log2_offset_delta": [0.0] * n,
        }
    )
    df.to_csv(path, index=False)


def test_cli_pressure_sfxi_multi_round(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    _write_records(records)

    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(
        campaign,
        workdir=workdir,
        records_path=records,
        model_params={"n_estimators": 5, "random_state": 0, "oob_score": False},
        objective_params={"setpoint_vector": [0, 0, 0, 1], "scaling": {"min_n": 1}},
        selection_params={"top_k": 2},
    )

    app = _build()
    runner = CliRunner()

    res = runner.invoke(app, ["--no-color", "init", "-c", str(campaign)])
    assert res.exit_code == 0, res.stdout

    # Round 0 labels
    labels0 = workdir / "labels_r0.csv"
    _write_sfxi_labels(labels0, seqs=["AAA", "BBB"])
    res = runner.invoke(
        app,
        ["--no-color", "ingest-y", "-c", str(campaign), "--round", "0", "--csv", str(labels0), "--yes"],
    )
    assert res.exit_code == 0, res.stdout

    res = runner.invoke(app, ["--no-color", "run", "-c", str(campaign), "--round", "0"])
    assert res.exit_code == 0, res.stdout

    # Round 1 labels
    labels1 = workdir / "labels_r1.csv"
    _write_sfxi_labels(labels1, seqs=["CCC"])
    res = runner.invoke(
        app,
        ["--no-color", "ingest-y", "-c", str(campaign), "--round", "1", "--csv", str(labels1), "--yes"],
    )
    assert res.exit_code == 0, res.stdout

    res = runner.invoke(app, ["--no-color", "run", "-c", str(campaign), "--round", "1"])
    assert res.exit_code == 0, res.stdout

    res = runner.invoke(app, ["--no-color", "status", "-c", str(campaign), "--with-ledger", "--json"])
    assert res.exit_code == 0, res.stdout
    status = json.loads(res.stdout)
    assert status["latest_round"]["round_index"] == 1

    res = runner.invoke(app, ["--no-color", "record-show", "-c", str(campaign), "--id", "e", "--json"])
    assert res.exit_code == 0, res.stdout
    report = json.loads(res.stdout)
    assert report.get("id") == "e"
    assert len(report.get("runs", [])) >= 2

    # Validate caches + label history on disk
    df = pd.read_parquet(records)
    store = RecordsStore(
        kind="local",
        records_path=records,
        campaign_slug="demo",
        x_col="X",
        y_col="Y",
        x_transform_name="identity",
        x_transform_params={},
    )
    lh = store.label_hist_col()
    hist_a = store._normalize_hist_cell(df.loc[df["id"] == "a", lh].iloc[0])
    hist_c = store._normalize_hist_cell(df.loc[df["id"] == "c", lh].iloc[0])
    assert any(e.get("r") == 0 for e in hist_a)
    assert any(e.get("r") == 1 for e in hist_c)

    col_r = store.latest_as_of_round_col()
    col_s = store.latest_pred_scalar_col()
    row_e = df.loc[df["id"] == "e", [col_r, col_s]].iloc[0]
    assert int(row_e[col_r]) == 1
    assert np.isfinite(float(row_e[col_s]))
