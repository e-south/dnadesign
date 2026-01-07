"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_cli_workflows.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from dnadesign.opal.src.cli.app import _build

from ._cli_helpers import write_campaign_yaml, write_records


def _setup_workspace(tmp_path: Path, *, include_opal_cols: bool = False) -> tuple[Path, Path, Path]:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    write_records(records, include_opal_cols=include_opal_cols)
    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(campaign, workdir=workdir, records_path=records)
    return workdir, campaign, records


def test_init_validate_explain_cli(tmp_path: Path) -> None:
    workdir, campaign, _ = _setup_workspace(tmp_path, include_opal_cols=True)
    app = _build()
    runner = CliRunner()

    res = runner.invoke(app, ["--no-color", "init", "-c", str(campaign)])
    assert res.exit_code == 0, res.stdout
    assert (workdir / "state.json").exists()
    assert (workdir / ".opal" / "config").exists()
    assert (workdir / "outputs").exists()
    assert (workdir / "inputs").exists()

    res = runner.invoke(app, ["--no-color", "validate", "-c", str(campaign)])
    assert res.exit_code == 0, res.stdout
    assert "validation passed" in res.stdout.lower()

    res = runner.invoke(app, ["--no-color", "explain", "-c", str(campaign), "--round", "0", "--json"])
    assert res.exit_code == 0, res.stdout
    out = json.loads(res.stdout)
    assert out["round_index"] == 0


def test_label_hist_validate_and_repair(tmp_path: Path) -> None:
    _, campaign, _ = _setup_workspace(tmp_path, include_opal_cols=True)
    app = _build()
    runner = CliRunner()

    res = runner.invoke(app, ["--no-color", "label-hist", "validate", "-c", str(campaign), "--json"])
    assert res.exit_code == 0, res.stdout
    out = json.loads(res.stdout)
    assert out["ok"] is True
    assert out["action"] == "validate"

    res = runner.invoke(app, ["--no-color", "label-hist", "repair", "-c", str(campaign), "--json"])
    assert res.exit_code == 0, res.stdout
    out = json.loads(res.stdout)
    assert out["ok"] is True
    assert out["action"] == "repair"
    assert out["applied"] is False


def test_ctx_show_audit_diff(tmp_path: Path) -> None:
    workdir, campaign, _ = _setup_workspace(tmp_path)
    app = _build()
    runner = CliRunner()

    round0 = workdir / "outputs" / "round_0"
    round1 = workdir / "outputs" / "round_1"
    round0.mkdir(parents=True, exist_ok=True)
    round1.mkdir(parents=True, exist_ok=True)

    ctx0 = {
        "core/run_id": "r0",
        "core/contracts/model/random_forest/produced": ["model/random_forest/x_dim"],
    }
    ctx1 = {
        "core/run_id": "r1",
        "core/contracts/model/random_forest/produced": [
            "model/random_forest/x_dim",
            "model/random_forest/y_dim",
        ],
    }
    (round0 / "round_ctx.json").write_text(json.dumps(ctx0))
    (round1 / "round_ctx.json").write_text(json.dumps(ctx1))

    res = runner.invoke(
        app,
        ["--no-color", "ctx", "show", "-c", str(campaign), "--round", "0", "--json"],
    )
    assert res.exit_code == 0, res.stdout
    out = json.loads(res.stdout)
    assert out["core/run_id"] == "r0"

    res = runner.invoke(
        app,
        ["--no-color", "ctx", "audit", "-c", str(campaign), "--round", "0", "--json"],
    )
    assert res.exit_code == 0, res.stdout
    audit = json.loads(res.stdout)
    assert "model" in audit
    assert "random_forest" in audit["model"]

    res = runner.invoke(
        app,
        [
            "--no-color",
            "ctx",
            "diff",
            "-c",
            str(campaign),
            "--round-a",
            "0",
            "--round-b",
            "1",
            "--json",
        ],
    )
    assert res.exit_code == 0, res.stdout
    diff = json.loads(res.stdout)
    assert "core/run_id" in diff.get("changed", {})


def test_ingest_y_cli(tmp_path: Path) -> None:
    workdir, campaign, _ = _setup_workspace(tmp_path)
    app = _build()
    runner = CliRunner()

    csv_path = workdir / "labels.csv"
    df = pd.DataFrame(
        {
            "sequence": ["AAA", "BBB"],
            "v00": [0.0, 0.0],
            "v10": [0.0, 0.0],
            "v01": [0.0, 0.0],
            "v11": [1.0, 0.5],
            "y00_star": [0.1, 0.2],
            "y10_star": [0.1, 0.2],
            "y01_star": [0.1, 0.2],
            "y11_star": [0.1, 0.2],
            "intensity_log2_offset_delta": [0.0, 0.0],
        }
    )
    df.to_csv(csv_path, index=False)

    res = runner.invoke(
        app,
        [
            "--no-color",
            "ingest-y",
            "-c",
            str(campaign),
            "--round",
            "0",
            "--csv",
            str(csv_path),
            "--yes",
        ],
    )
    assert res.exit_code == 0, res.stdout
    assert (workdir / "outputs" / "ledger.labels.parquet").exists()


def test_ingest_y_rejects_unsupported_extension(tmp_path: Path) -> None:
    workdir, campaign, _ = _setup_workspace(tmp_path)
    app = _build()
    runner = CliRunner()

    bad_path = workdir / "labels.txt"
    df = pd.DataFrame(
        {
            "sequence": ["AAA"],
            "v00": [0.0],
            "v10": [0.0],
            "v01": [0.0],
            "v11": [1.0],
            "y00_star": [0.1],
            "y10_star": [0.1],
            "y01_star": [0.1],
            "y11_star": [0.1],
            "intensity_log2_offset_delta": [0.0],
        }
    )
    df.to_csv(bad_path, index=False)

    res = runner.invoke(
        app,
        [
            "--no-color",
            "ingest-y",
            "-c",
            str(campaign),
            "--round",
            "0",
            "--csv",
            str(bad_path),
            "--yes",
        ],
    )
    assert res.exit_code != 0, res.stdout
    assert "must be a CSV or Parquet file" in res.output


def test_ingest_y_rejects_params_non_json(tmp_path: Path) -> None:
    workdir, campaign, _ = _setup_workspace(tmp_path)
    app = _build()
    runner = CliRunner()

    csv_path = workdir / "labels.csv"
    df = pd.DataFrame(
        {
            "sequence": ["AAA"],
            "v00": [0.0],
            "v10": [0.0],
            "v01": [0.0],
            "v11": [1.0],
            "y00_star": [0.1],
            "y10_star": [0.1],
            "y01_star": [0.1],
            "y11_star": [0.1],
            "intensity_log2_offset_delta": [0.0],
        }
    )
    df.to_csv(csv_path, index=False)

    bad_params = workdir / "params.txt"
    bad_params.write_text("{}")

    res = runner.invoke(
        app,
        [
            "--no-color",
            "ingest-y",
            "-c",
            str(campaign),
            "--round",
            "0",
            "--csv",
            str(csv_path),
            "--params",
            str(bad_params),
            "--yes",
        ],
    )
    assert res.exit_code != 0, res.stdout
    assert "must be a JSON file" in res.output
