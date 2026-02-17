"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_cli_notebook_generate.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

from typer.testing import CliRunner

from dnadesign.opal.src.cli.app import _build

from ._cli_helpers import (
    write_campaign_yaml,
    write_ledger,
    write_ledger_labels,
    write_records,
)


def test_notebook_generate_smoke(tmp_path: Path) -> None:
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

    out_path = workdir / "notebooks" / "opal_demo_analysis.py"
    app = _build()
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "--no-color",
            "notebook",
            "generate",
            "-c",
            str(campaign),
            "--out",
            str(out_path),
            "--no-validate",
        ],
    )
    assert res.exit_code == 0, res.stdout
    assert out_path.exists()

    txt = out_path.read_text()
    assert "marimo.App" in txt
    assert "CampaignAnalysis.from_config_path" in txt
    assert "opal" in txt.lower()
    assert "Data source" in txt
    assert "mo.ui.table" in txt
    assert "__generated_with" in txt
    assert 'marimo.App(width="medium")' in txt

    # Optional import check if marimo is installed
    if importlib.util.find_spec("marimo") is not None:
        spec = importlib.util.spec_from_file_location("opal_campaign_nb", out_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert hasattr(module, "app")


def test_notebook_generate_requires_ledger_by_default(tmp_path: Path) -> None:
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

    app = _build()
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "--no-color",
            "notebook",
            "generate",
            "-c",
            str(campaign),
        ],
    )
    assert res.exit_code != 0, res.output
    assert "Missing runs sink" in res.output


def test_notebook_generate_rejects_unknown_round(tmp_path: Path) -> None:
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
    write_ledger(workdir, run_id="run-0", round_index=0)
    write_ledger_labels(workdir, round_index=0)

    app = _build()
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "--no-color",
            "notebook",
            "generate",
            "-c",
            str(campaign),
            "--round",
            "7",
        ],
    )
    assert res.exit_code != 0, res.output
    assert "Available rounds" in res.output


def test_notebook_generate_with_name(tmp_path: Path) -> None:
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

    app = _build()
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "--no-color",
            "notebook",
            "generate",
            "-c",
            str(campaign),
            "--name",
            "custom_demo",
            "--no-validate",
        ],
    )
    assert res.exit_code == 0, res.stdout
    out_path = workdir / "notebooks" / "custom_demo.py"
    assert out_path.exists()


def test_notebook_generate_existing_name_requires_force(tmp_path: Path) -> None:
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

    existing = workdir / "notebooks" / "opal_demo_analysis.py"
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_text("import marimo\n")

    app = _build()
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "--no-color",
            "notebook",
            "generate",
            "-c",
            str(campaign),
            "--no-validate",
        ],
    )
    assert res.exit_code != 0, res.output
    lowered = res.output.lower()
    assert "already exists" in lowered
    assert "--force" in lowered
    assert "--name" in lowered


def test_notebook_run_selects_single_notebook(tmp_path: Path, monkeypatch) -> None:
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

    nb_path = workdir / "notebooks" / "only.py"
    nb_path.parent.mkdir(parents=True, exist_ok=True)
    nb_path.write_text("import marimo\n")

    import dnadesign.opal.src.cli.commands.notebook as notebook_cmd

    monkeypatch.setattr(notebook_cmd.importlib.util, "find_spec", lambda name: object())
    calls: list[list[str]] = []

    def _fake_run(args, check):
        _unused = check
        calls.append(list(args))
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(notebook_cmd.subprocess, "run", _fake_run)

    app = _build()
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "--no-color",
            "notebook",
            "run",
            "-c",
            str(campaign),
        ],
    )
    assert res.exit_code == 0, res.stdout
    assert calls
    assert str(nb_path) in calls[0]


def test_notebook_run_requires_notebook(tmp_path: Path) -> None:
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

    app = _build()
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "--no-color",
            "notebook",
            "run",
            "-c",
            str(campaign),
        ],
    )
    assert res.exit_code != 0, res.output
    assert "no notebooks found" in res.output.lower()


def test_notebook_run_multiple_notebooks_non_tty(tmp_path: Path) -> None:
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

    nb_dir = workdir / "notebooks"
    nb_dir.mkdir(parents=True, exist_ok=True)
    (nb_dir / "one.py").write_text("import marimo\n")
    (nb_dir / "two.py").write_text("import marimo\n")

    app = _build()
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "--no-color",
            "notebook",
            "run",
            "-c",
            str(campaign),
        ],
    )
    assert res.exit_code != 0, res.output
    lowered = res.output.lower()
    assert "multiple notebooks found" in lowered
    assert "0:" in lowered
    assert "1:" in lowered
    assert "--path" in lowered


def test_notebook_generate_requires_records_even_no_validate(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(
        campaign,
        workdir=workdir,
        records_path=records,
    )
    write_ledger(workdir, run_id="run-0", round_index=0)
    write_ledger_labels(workdir, round_index=0)

    app = _build()
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "--no-color",
            "notebook",
            "generate",
            "-c",
            str(campaign),
            "--no-validate",
        ],
    )
    assert res.exit_code != 0, res.output
    assert "records.parquet not found" in res.output


def test_notebook_root_rich_output(tmp_path: Path, monkeypatch) -> None:
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

    nb_dir = workdir / "notebooks"
    nb_dir.mkdir(parents=True, exist_ok=True)
    (nb_dir / "one.py").write_text("import marimo\n")
    (nb_dir / "two.py").write_text("import marimo\n")

    import dnadesign.opal.src.cli.commands.notebook as notebook_cmd

    calls: list[object] = []

    monkeypatch.setattr(notebook_cmd, "tui_enabled", lambda: True)
    monkeypatch.setattr(
        notebook_cmd,
        "_print_rich",
        lambda obj: calls.append(obj) or True,
    )

    app = _build()
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "--no-color",
            "notebook",
            "-c",
            str(campaign),
        ],
    )
    assert res.exit_code == 0, res.output
    assert calls


def test_notebook_rich_tables_use_rounded_box() -> None:
    from rich import box

    import dnadesign.opal.src.cli.commands.notebook as notebook_cmd

    kv_table = notebook_cmd._rich_kv_table("Notebook", {"Key": "Value"})
    list_table = notebook_cmd._rich_list_table("Notebooks", ["0: one.py"])

    assert kv_table.box == box.ROUNDED
    assert list_table.box == box.ROUNDED
    assert str(kv_table.border_style) == "cyan"
    assert str(list_table.border_style) == "cyan"
