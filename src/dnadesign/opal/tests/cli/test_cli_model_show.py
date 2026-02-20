"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_cli_model_show.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import json
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from dnadesign.opal.src.cli.app import _build
from dnadesign.opal.src.registries.models import get_model
from dnadesign.opal.tests._cli_helpers import (
    write_campaign_yaml,
    write_records,
    write_state,
)


def test_model_show_with_model_meta(tmp_path):
    round_dir = tmp_path / "outputs" / "rounds" / "round_0"
    model_dir = round_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.joblib"

    model = get_model("random_forest", {"n_estimators": 5, "random_state": 0, "oob_score": False})
    X = np.array([[0.1, 0.2], [0.2, 0.3]])
    Y = np.array([[0.4], [0.6]])
    model.fit(X, Y)
    model.save(str(model_path))

    meta = {
        "model__name": "random_forest",
        "model__params": {"n_estimators": 5, "random_state": 0},
    }
    (model_dir / "model_meta.json").write_text(json.dumps(meta))

    app = _build()
    runner = CliRunner()
    res = runner.invoke(app, ["--no-color", "model-show", "--model-path", str(model_path), "--json"])
    assert res.exit_code == 0, res.stdout
    out = json.loads(res.stdout)
    assert out["model_type"] == "random_forest"
    assert int(out["params"]["n_estimators"]) == 5


def test_model_show_round_latest(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    write_records(records)
    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(
        campaign,
        workdir=workdir,
        records_path=records,
    )

    round_dir = workdir / "outputs" / "rounds" / "round_0"
    model_dir = round_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.joblib"
    model = get_model("random_forest", {"n_estimators": 5, "random_state": 0, "oob_score": False})
    X = np.array([[0.1, 0.2], [0.2, 0.3]])
    Y = np.array([[0.4], [0.6]])
    model.fit(X, Y)
    model.save(str(model_path))
    meta = {
        "model__name": "random_forest",
        "model__params": {"n_estimators": 5, "random_state": 0},
    }
    (model_dir / "model_meta.json").write_text(json.dumps(meta))
    write_state(workdir, records_path=records, run_id="r0-test", round_index=0)

    app = _build()
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "--no-color",
            "model-show",
            "--config",
            str(campaign),
            "--round",
            "latest",
            "--json",
        ],
    )
    assert res.exit_code == 0, res.stdout
    out = json.loads(res.stdout)
    assert out["model_type"] == "random_forest"
