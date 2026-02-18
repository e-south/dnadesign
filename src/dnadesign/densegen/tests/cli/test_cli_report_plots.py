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
    expected_literals = [
        'app = marimo.App(width="medium")',
        "workspace-scoped run dashboard for DenseGen outputs",
        "from dnadesign.baserender import load_records_from_parquet",
        "from dnadesign.baserender import render_record_figure",
        "from dnadesign.densegen.src.viz.plot_registry import PLOT_SPECS",
        (
            "from dnadesign.densegen.src.integrations.baserender.notebook_contract import "
            "densegen_notebook_render_contract"
        ),
        "record_window_limit = int(contract.record_window_limit)",
        "Failed to parse `run_manifest.json`",
        "Failed to parse `plot_manifest.json`",
        'contract.adapter_columns["id"]',
        'contract.adapter_columns["sequence"]',
        'contract.adapter_columns["annotations"]',
        "Duplicate record ids detected in the notebook preview window",
        'record_plan_filter = mo.ui.dropdown(options=_plan_options, value=_plan_options[0], label="Record plan")',
        "prev_record_button = mo.ui.button(",
        "next_record_button = mo.ui.button(",
        'export_format = mo.ui.dropdown(options=["parquet", "csv"], value="parquet", label="Export format")',
        'export_path = mo.ui.text(value=str(default_export_path), label="Export path", full_width=True)',
        'raise RuntimeError(f"Export failed while writing `{destination}`: {exc}")',
        'plot_selector = mo.ui.dropdown(options=plot_options, value=plot_options[0], label="Plot")',
        'selected_plot_plan = str(selected_record_plan or "all")',
        '_allowed_plans = {selected_plot_plan, "unscoped", "stage_a"}',
        'mo.pdf(_plot_path, width="100%", height=f"{int(plot_height_px.value)}px")',
        'if _suffix == ".pdf":',
        '_suffix in {".png", ".jpg", ".jpeg", ".svg", ".webp", ".gif"}',
        'baserender_figure.patch.set_facecolor("white")',
    ]
    for literal in expected_literals:
        assert literal in content

    absent_literals = [
        "__RUN_ROOT__",
        "__CFG_PATH__",
        "__RECORDS_PATH__",
        "__OUTPUT_SOURCE__",
        "__USR_ROOT__",
        "__USR_DATASET__",
        "Use **Refresh**",
        'refresh = mo.ui.run_button(label="Refresh", kind="neutral")',
        "mo.stop(",
        "dense_arrays.parquet",
        'status_message = ""',
        'status_message = f"Export failed',
    ]
    for literal in absent_literals:
        assert literal not in content


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
    assert "__OUTPUT_SOURCE__" not in content
    assert "__USR_ROOT__" not in content
    assert "__USR_DATASET__" not in content
    assert 'output_source = "usr"' in content
    assert '"records_with_overlays.parquet"' in content


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
    assert 'output_source = "usr"' in content
    assert '"records_with_overlays.parquet"' in content


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


def test_notebook_run_uses_marimo_edit_mode_by_default(tmp_path: Path, monkeypatch) -> None:
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
    assert captured["command"][:2] == ["marimo", "edit"]
    assert "--headless" not in captured["command"]
    assert captured["check"] is True
    env = captured["env"]
    assert isinstance(env, dict)
    assert env.get("MARIMO_SKIP_UPDATE_CHECK") == "1"
    assert "BROWSER" not in env
    assert "Notebook URL" not in run_result.output


def test_notebook_run_explicit_open_keeps_marimo_auto_open_behavior(tmp_path: Path, monkeypatch) -> None:
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
        captured["env"] = env
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", _fake_run)
    run_result = runner.invoke(app, ["notebook", "run", "--mode", "run", "--open", "-c", str(cfg_path)])

    assert run_result.exit_code == 0, run_result.output
    assert captured["command"][:2] == ["marimo", "run"]
    assert "--headless" not in captured["command"]
    assert "Notebook URL" in run_result.output


def test_notebook_run_no_open_passes_headless_to_marimo(tmp_path: Path, monkeypatch) -> None:
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
        captured["env"] = env
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", _fake_run)
    run_result = runner.invoke(app, ["notebook", "run", "--mode", "run", "--no-open", "-c", str(cfg_path)])

    assert run_result.exit_code == 0, run_result.output
    assert captured["command"][:2] == ["marimo", "run"]
    assert "--headless" in captured["command"]
    env = captured.get("env")
    assert isinstance(env, dict)
    assert "BROWSER" not in env
    assert "Notebook URL" in run_result.output


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
    run_result = runner.invoke(app, ["notebook", "run", "--mode", "run", "--headless", "-c", str(cfg_path)])
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


def test_notebook_run_rejects_no_open_edit_mode(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    gen_result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert gen_result.exit_code == 0, gen_result.output

    import dnadesign.densegen.src.cli.notebook as notebook_commands

    monkeypatch.setattr(notebook_commands, "_ensure_marimo_installed", lambda: None)
    run_result = runner.invoke(app, ["notebook", "run", "--mode", "edit", "--no-open", "-c", str(cfg_path)])
    assert run_result.exit_code == 1
    assert "--open/--no-open is only supported with --mode run" in run_result.output


def test_notebook_run_rejects_invalid_host(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    gen_result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert gen_result.exit_code == 0, gen_result.output

    import dnadesign.densegen.src.cli.notebook as notebook_commands

    monkeypatch.setattr(notebook_commands, "_ensure_marimo_installed", lambda: None)
    run_result = runner.invoke(app, ["notebook", "run", "--host", "   ", "-c", str(cfg_path)])
    assert run_result.exit_code == 1
    assert "--host must be a non-empty value" in run_result.output


def test_notebook_run_rejects_invalid_port(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    gen_result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert gen_result.exit_code == 0, gen_result.output

    import dnadesign.densegen.src.cli.notebook as notebook_commands

    monkeypatch.setattr(notebook_commands, "_ensure_marimo_installed", lambda: None)
    run_result = runner.invoke(app, ["notebook", "run", "--port", "70000", "-c", str(cfg_path)])
    assert run_result.exit_code == 1
    assert "--port must be within 1-65535" in run_result.output


def test_notebook_run_strips_host_before_launch(tmp_path: Path, monkeypatch) -> None:
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
    run_result = runner.invoke(app, ["notebook", "run", "--host", " 127.0.0.1 ", "-c", str(cfg_path)])

    assert run_result.exit_code == 0, run_result.output
    assert captured["command"][:2] == ["marimo", "edit"]
    assert "--host" in captured["command"]
    host_index = captured["command"].index("--host")
    assert captured["command"][host_index + 1] == "127.0.0.1"
    assert "http://127.0.0.1:2718" not in run_result.output


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
