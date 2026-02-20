"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_label_hist_attach_from_y.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from dnadesign.opal.src.cli.app import _build
from dnadesign.opal.tests._cli_helpers import write_campaign_yaml


def _write_records(path: Path) -> None:
    df = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "sequence": ["AAA", "BBB", "CCC", "DDD"],
            "bio_type": ["dna"] * 4,
            "alphabet": ["dna_4"] * 4,
            "X": [[0.1], [0.2], [0.3], [0.4]],
            "Y": [[0.1], [0.2], None, None],
        }
    )
    df.to_parquet(path, index=False)


def test_label_hist_attach_from_y(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    _write_records(records)

    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(
        campaign,
        workdir=workdir,
        records_path=records,
        transforms_y_name="scalar_from_table_v1",
        transforms_y_params={"sequence_column": "sequence", "y_column": "y"},
        objective_name="scalar_identity_v1",
        objective_params={},
        y_expected_length=1,
        model_params={"n_estimators": 5, "random_state": 0, "oob_score": False},
        selection_params={"top_k": 1},
    )

    app = _build()
    runner = CliRunner()

    res = runner.invoke(app, ["--no-color", "init", "-c", str(campaign)])
    assert res.exit_code == 0, res.stdout

    # Run should fail until label_hist is explicitly attached from existing Y.
    res = runner.invoke(app, ["--no-color", "run", "-c", str(campaign), "--round", "0"])
    assert res.exit_code != 0

    res = runner.invoke(
        app,
        [
            "--no-color",
            "label-hist",
            "attach-from-y",
            "-c",
            str(campaign),
            "--round",
            "0",
            "--apply",
        ],
    )
    assert res.exit_code == 0, res.stdout

    res = runner.invoke(app, ["--no-color", "run", "-c", str(campaign), "--round", "0"])
    assert res.exit_code == 0, res.stdout
