"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/tests/cli/test_json_outputs.py

Machine-readable CLI output contracts.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from dnadesign.permuter.src.cli.app import app


def test_run_evaluate_validate_and_plot_emit_json(tmp_path: Path) -> None:
    workspace = _toy_workspace(tmp_path)
    runner = CliRunner()

    run = runner.invoke(app, ["run", "--workspace", str(workspace), "--ref", "toy", "--json"])
    assert run.exit_code == 0, run.output
    run_payload = json.loads(run.output)
    assert run_payload["schema"] == "permuter.run.v1"
    assert run_payload["row_count"] == 9

    evaluate = runner.invoke(
        app,
        [
            "evaluate",
            "--workspace",
            str(workspace),
            "--ref",
            "toy",
            "--with",
            "smoke:placeholder:log_likelihood",
            "--json",
        ],
    )
    assert evaluate.exit_code == 0, evaluate.output
    evaluate_payload = json.loads(evaluate.output)
    assert evaluate_payload["schema"] == "permuter.evaluate.v1"
    assert evaluate_payload["metrics"] == ["smoke"]

    validate = runner.invoke(app, ["validate", "--data", str(workspace / "outputs"), "--strict", "--json"])
    assert validate.exit_code == 0, validate.output
    validate_payload = json.loads(validate.output)
    assert validate_payload["schema"] == "permuter.validate.v1"
    assert validate_payload["metric_ids"] == ["smoke"]

    plot = runner.invoke(
        app,
        [
            "plot",
            "--data",
            str(workspace / "outputs"),
            "--metric-id",
            "smoke",
            "--which",
            "metric_by_mutation_count",
            "--no-emit-summaries",
            "--json",
        ],
    )
    assert plot.exit_code == 0, plot.output
    plot_payload = json.loads(plot.output)
    assert plot_payload["schema"] == "permuter.plot.v1"
    assert plot_payload["artifacts"][0]["id"] == "metric_by_mutation_count"
    assert Path(plot_payload["manifest"]).exists()


def test_plot_rejects_unsupported_ids_without_traceback(tmp_path: Path) -> None:
    workspace = _toy_workspace(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, ["run", "--workspace", str(workspace), "--ref", "toy"])
    assert result.exit_code == 0, result.output
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--workspace",
            str(workspace),
            "--ref",
            "toy",
            "--with",
            "smoke:placeholder:log_likelihood",
        ],
    )
    assert result.exit_code == 0, result.output

    bad = runner.invoke(
        app,
        [
            "plot",
            "--data",
            str(workspace / "outputs"),
            "--metric-id",
            "smoke",
            "--which",
            "window_score_mass",
        ],
    )

    assert bad.exit_code != 0
    assert "Unknown plot" in bad.output
    assert "Traceback" not in bad.output


def _toy_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "toy"
    workspace.mkdir(parents=True)
    workspace.joinpath("refs.csv").write_text("ref_name,sequence\ntoy,ACG\n", encoding="utf-8")
    workspace.joinpath("config.yaml").write_text(
        """
scope:
  name: toy
  bio_type: dna
  input:
    refs: "${WORKSPACE_DIR}/refs.csv"
    name_col: ref_name
    seq_col: sequence
  permute:
    protocol: scan_dna
    params: {}
  output:
    dir: "${WORKSPACE_DIR}/outputs"
    layout: flat
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return workspace
