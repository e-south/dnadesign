"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_cli_predict_yops_inversion.py

Module Author(s): Eric J. South (extended by Codex)
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from dnadesign.opal.src.cli.app import _build
from dnadesign.opal.src.models.random_forest import RandomForestModel

from ._cli_helpers import write_campaign_yaml, write_records, write_state


def _train_model(model_path: Path) -> dict:
    X_train = np.array([[0.1, 0.2], [0.2, 0.3]])
    Y_train = np.array([[1.0], [2.0]])
    model = RandomForestModel(params={"n_estimators": 5, "random_state": 1, "bootstrap": True, "oob_score": False})
    model.fit(X_train, Y_train)
    model.save(str(model_path))
    return {"model__name": "random_forest", "model__params": model.get_params(), "x_dim": 2, "y_dim": 1}


def test_predict_requires_round_ctx_when_yops_present(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    write_records(records)
    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(campaign, workdir=workdir, records_path=records)

    round_dir = workdir / "outputs" / "round_0"
    round_dir.mkdir(parents=True, exist_ok=True)
    model_path = round_dir / "model.joblib"
    meta = _train_model(model_path)
    meta["training__y_ops"] = [{"name": "intensity_median_iqr", "params": {"min_labels": 1}}]
    (round_dir / "model_meta.json").write_text(json.dumps(meta))

    app = _build()
    runner = CliRunner()
    res = runner.invoke(app, ["--no-color", "predict", "-c", str(campaign), "--model-path", str(model_path)])
    assert res.exit_code != 0
    assert "round_ctx.json is missing" in res.output

    res = runner.invoke(
        app,
        ["--no-color", "predict", "-c", str(campaign), "--model-path", str(model_path), "--assume-no-yops"],
    )
    assert res.exit_code == 0, res.output
    assert "y_pred_vec" in res.output


def test_predict_accepts_latest_round_selector(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    write_records(records)
    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(campaign, workdir=workdir, records_path=records)

    round_dir = workdir / "outputs" / "round_0"
    round_dir.mkdir(parents=True, exist_ok=True)
    model_path = round_dir / "model.joblib"
    meta = _train_model(model_path)
    (round_dir / "model_meta.json").write_text(json.dumps(meta))

    write_state(workdir, records_path=records, run_id="run-0", round_index=0)

    app = _build()
    runner = CliRunner()
    res = runner.invoke(app, ["--no-color", "predict", "-c", str(campaign), "--round", "latest"])
    assert res.exit_code == 0, res.output
    assert "y_pred_vec" in res.output


def test_predict_rejects_round_and_model_path(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    write_records(records)
    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(campaign, workdir=workdir, records_path=records)

    model_path = workdir / "outputs" / "round_0" / "model.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    _train_model(model_path)

    app = _build()
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "--no-color",
            "predict",
            "-c",
            str(campaign),
            "--round",
            "0",
            "--model-path",
            str(model_path),
        ],
    )
    assert res.exit_code != 0, res.output
    assert "mutually exclusive" in res.output.lower()


def test_predict_errors_on_missing_model_path(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    write_records(records)
    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(campaign, workdir=workdir, records_path=records)

    missing_path = workdir / "outputs" / "round_0" / "missing.joblib"

    app = _build()
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "--no-color",
            "predict",
            "-c",
            str(campaign),
            "--model-path",
            str(missing_path),
        ],
    )
    assert res.exit_code != 0, res.output
    assert "--model-path not found" in res.output


def test_predict_rejects_unsupported_out_extension(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    write_records(records)
    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(campaign, workdir=workdir, records_path=records)

    round_dir = workdir / "outputs" / "round_0"
    round_dir.mkdir(parents=True, exist_ok=True)
    model_path = round_dir / "model.joblib"
    meta = _train_model(model_path)
    (round_dir / "model_meta.json").write_text(json.dumps(meta))

    write_state(workdir, records_path=records, run_id="run-0", round_index=0)

    out_path = workdir / "preds.txt"

    app = _build()
    runner = CliRunner()
    res = runner.invoke(
        app,
        ["--no-color", "predict", "-c", str(campaign), "--round", "latest", "--out", str(out_path)],
    )
    assert res.exit_code != 0, res.output
    assert "must be a CSV or Parquet file" in res.output


def test_predict_rejects_model_params_non_json(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    write_records(records)
    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(campaign, workdir=workdir, records_path=records)

    round_dir = workdir / "outputs" / "round_0"
    round_dir.mkdir(parents=True, exist_ok=True)
    model_path = round_dir / "model.joblib"
    _train_model(model_path)

    bad_params = workdir / "params.txt"
    bad_params.write_text("{}")

    app = _build()
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "--no-color",
            "predict",
            "-c",
            str(campaign),
            "--model-path",
            str(model_path),
            "--model-name",
            "random_forest",
            "--model-params",
            str(bad_params),
        ],
    )
    assert res.exit_code != 0, res.output
    assert "must be a JSON file" in res.output


def test_predict_errors_on_x_dim_mismatch(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    write_records(records)
    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(campaign, workdir=workdir, records_path=records)

    round_dir = workdir / "outputs" / "round_0"
    round_dir.mkdir(parents=True, exist_ok=True)
    model_path = round_dir / "model.joblib"
    meta = _train_model(model_path)
    meta["x_dim"] = 3
    (round_dir / "model_meta.json").write_text(json.dumps(meta))

    app = _build()
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "--no-color",
            "predict",
            "-c",
            str(campaign),
            "--model-path",
            str(model_path),
        ],
    )
    assert res.exit_code != 0, res.output
    assert "X dimension mismatch" in res.output
