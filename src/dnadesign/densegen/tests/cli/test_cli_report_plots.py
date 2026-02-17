"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/cli/test_cli_report_plots.py

CLI guardrails for DenseGen notebook command surfaces.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

from typer.testing import CliRunner

from dnadesign.densegen.src.cli.main import app
from dnadesign.densegen.tests.config_fixtures import write_minimal_config


def test_report_command_removed() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["report", "--help"])
    assert result.exit_code != 0
    assert "No such command 'report'" in result.output


def test_notebook_generate_writes_workspace_notebook(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert result.exit_code == 0, result.output
    notebook_path = tmp_path / "outputs" / "notebooks" / "densegen_run_overview.py"
    assert notebook_path.exists()
    content = notebook_path.read_text()
    assert "DenseGen Run Notebook" in content
    assert "from dnadesign.baserender import load_records_from_parquet" in content
    assert "from dnadesign.baserender import render_record_figure" in content
    assert (
        "from dnadesign.densegen.src.integrations.baserender.notebook_contract import densegen_notebook_render_contract"
    ) in content
    assert "records_path = Path(" in content
    assert "dense_arrays.parquet" not in content
    assert "record_window_limit = int(contract.record_window_limit)" in content
    assert "ParquetFile(records_path)" in content
    assert "pd.read_parquet(records_path)" not in content
    assert "required = {" in content
    assert 'contract.adapter_columns["id"]' in content
    assert 'contract.adapter_columns["sequence"]' in content
    assert 'contract.adapter_columns["annotations"]' in content
    assert 'required = {"id", "sequence", "densegen__used_tfbs_detail"}' not in content
    assert "duplicate_id_count = int(df_window[record_id_column].astype(str).duplicated().sum())" in content
    assert "Duplicate ids detected in records preview window" in content


def test_notebook_generate_uses_configured_parquet_path(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    cfg_path.write_text(
        cfg_path.read_text().replace("outputs/tables/records.parquet", "outputs/tables/custom_records.parquet")
    )
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert result.exit_code == 0, result.output
    notebook_path = tmp_path / "outputs" / "notebooks" / "densegen_run_overview.py"
    content = notebook_path.read_text()
    assert "custom_records.parquet" in content
    assert "No `records.parquet` artifact was found for this workspace" not in content
    assert "Ensure `records.parquet` has unique `id` values" not in content


def test_notebook_generate_supports_usr_output_target(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            """
            densegen:
              schema_version: "2.9"
              run:
                id: demo
                root: "."
              inputs:
                - name: demo_input
                  type: binding_sites
                  path: inputs.csv
              output:
                targets: [usr]
                schema:
                  bio_type: dna
                  alphabet: dna_4
                usr:
                  dataset: demo
                  root: outputs/usr_datasets
              generation:
                sequence_length: 10
                plan:
                  - name: demo_plan
                    quota: 1
                    sampling:
                      include_inputs: [demo_input]
                    regulator_constraints:
                      groups: []
              solver:
                backend: CBC
                strategy: iterate
              logging:
                log_dir: outputs/logs
            """
        ).strip()
        + "\n"
    )
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert result.exit_code == 0, result.output
    notebook_path = tmp_path / "outputs" / "notebooks" / "densegen_run_overview.py"
    content = notebook_path.read_text()
    assert "outputs/usr_datasets/demo/records.parquet" in content


def test_notebook_generate_uses_plots_source_when_output_targets_are_both(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            """
            densegen:
              schema_version: "2.9"
              run:
                id: demo
                root: "."
              inputs:
                - name: demo_input
                  type: binding_sites
                  path: inputs.csv
              output:
                targets: [parquet, usr]
                schema:
                  bio_type: dna
                  alphabet: dna_4
                parquet:
                  path: outputs/tables/records.parquet
                usr:
                  dataset: demo
                  root: outputs/usr_datasets
              generation:
                sequence_length: 10
                plan:
                  - name: demo_plan
                    quota: 1
                    sampling:
                      include_inputs: [demo_input]
                    regulator_constraints:
                      groups: []
              solver:
                backend: CBC
                strategy: iterate
              logging:
                log_dir: outputs/logs
            plots:
              source: usr
              out_dir: outputs/plots
            """
        ).strip()
        + "\n"
    )
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert result.exit_code == 0, result.output
    notebook_path = tmp_path / "outputs" / "notebooks" / "densegen_run_overview.py"
    content = notebook_path.read_text()
    assert "outputs/usr_datasets/demo/records.parquet" in content


def test_notebook_generate_custom_out_suggests_run_with_path(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")
    out_path = tmp_path / "custom notebook.py"

    runner = CliRunner()
    result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path), "--out", str(out_path)])
    assert result.exit_code == 0, result.output
    assert "dense notebook run --path 'custom notebook.py'" in result.output


def test_notebook_run_requires_existing_notebook(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    result = runner.invoke(app, ["notebook", "run", "-c", str(cfg_path)])
    assert result.exit_code == 1
    assert "No notebook found" in result.output
    assert "outputs/notebooks/densegen_run_overview.py" in result.output


def test_notebook_generate_passes_marimo_check(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert result.exit_code == 0, result.output
    notebook_path = tmp_path / "outputs" / "notebooks" / "densegen_run_overview.py"
    assert notebook_path.exists()

    check_result = subprocess.run(
        [sys.executable, "-m", "marimo", "check", str(notebook_path)],
        capture_output=True,
        text=True,
    )
    assert check_result.returncode == 0, check_result.stdout + check_result.stderr
    assert "warning[" not in check_result.stdout


def test_notebook_run_uses_marimo_run_mode_by_default(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    gen_result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert gen_result.exit_code == 0, gen_result.output

    import dnadesign.densegen.src.cli.notebook as notebook_commands

    monkeypatch.setattr(notebook_commands, "_ensure_marimo_installed", lambda: None)
    captured: dict[str, object] = {}

    def _fake_run(command, check, env):
        captured["command"] = command
        captured["check"] = check
        captured["env"] = env
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", _fake_run)
    run_result = runner.invoke(app, ["notebook", "run", "-c", str(cfg_path)])
    assert run_result.exit_code == 0, run_result.output
    assert captured["command"][:2] == ["marimo", "run"]
    assert captured["check"] is True
    env = captured["env"]
    assert isinstance(env, dict)
    assert env.get("MARIMO_SKIP_UPDATE_CHECK") == "1"


def test_notebook_run_edit_mode_failure_suggests_run_mode(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    gen_result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert gen_result.exit_code == 0, gen_result.output

    import dnadesign.densegen.src.cli.notebook as notebook_commands

    monkeypatch.setattr(notebook_commands, "_ensure_marimo_installed", lambda: None)

    def _fail_run(command, check, env):
        raise subprocess.CalledProcessError(returncode=1, cmd=command)

    monkeypatch.setattr(subprocess, "run", _fail_run)
    run_result = runner.invoke(app, ["notebook", "run", "--mode", "edit", "-c", str(cfg_path)])
    assert run_result.exit_code == 1
    assert "dense notebook run --mode run" in run_result.output


def test_notebook_run_headless_passes_flag_to_marimo(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    gen_result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert gen_result.exit_code == 0, gen_result.output

    import dnadesign.densegen.src.cli.notebook as notebook_commands

    monkeypatch.setattr(notebook_commands, "_ensure_marimo_installed", lambda: None)
    captured: dict[str, object] = {}

    def _fake_run(command, check, env):
        captured["command"] = command
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", _fake_run)
    run_result = runner.invoke(app, ["notebook", "run", "--headless", "-c", str(cfg_path)])
    assert run_result.exit_code == 0, run_result.output
    assert captured["command"][:2] == ["marimo", "run"]
    assert "--headless" in captured["command"]


def test_notebook_run_rejects_headless_edit_mode(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    gen_result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert gen_result.exit_code == 0, gen_result.output

    import dnadesign.densegen.src.cli.notebook as notebook_commands

    monkeypatch.setattr(notebook_commands, "_ensure_marimo_installed", lambda: None)
    run_result = runner.invoke(app, ["notebook", "run", "--mode", "edit", "--headless", "-c", str(cfg_path)])
    assert run_result.exit_code == 1
    assert "--headless is only supported with --mode run" in run_result.output


def test_plot_missing_records_reports_actionable_error(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    result = runner.invoke(app, ["plot", "-c", str(cfg_path)])

    assert result.exit_code == 1
    assert "Plot generation failed" in result.output
    assert "Parquet output not found" in result.output
    assert "dense run --fresh --no-plot" in result.output


def test_plot_missing_records_with_dual_sinks_reports_plots_source_hint(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            """
            densegen:
              schema_version: "2.9"
              run:
                id: demo
                root: "."
              inputs:
                - name: demo_input
                  type: binding_sites
                  path: inputs.csv
              output:
                targets: [parquet, usr]
                schema:
                  bio_type: dna
                  alphabet: dna_4
                parquet:
                  path: outputs/tables/records.parquet
                usr:
                  root: outputs/usr_datasets
                  dataset: densegen/demo
              generation:
                sequence_length: 10
                plan:
                  - name: demo_plan
                    quota: 1
                    sampling:
                      include_inputs: [demo_input]
                    regulator_constraints:
                      groups: []
              solver:
                backend: CBC
                strategy: iterate
              logging:
                log_dir: outputs/logs
            plots:
              source: parquet
              out_dir: outputs/plots
              default: [placement_map]
            """
        ).strip()
        + "\n"
    )
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    result = runner.invoke(app, ["plot", "-c", str(cfg_path)])

    assert result.exit_code == 1
    assert "Plot generation failed" in result.output
    assert "plots.source" in result.output
