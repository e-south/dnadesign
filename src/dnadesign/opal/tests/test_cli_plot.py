"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_cli_plot.py

CLI integration tests for plot command.

Module Author(s): Eric J. South (extended by Codex)
Dunlop Lab
--------------------------------------------------------------------------------
"""

from pathlib import Path

from typer.testing import CliRunner

from dnadesign.opal.src.cli.app import _build
from dnadesign.opal.src.plots._context import PlotContext
from dnadesign.opal.src.registries.plot import register_plot

from ._cli_helpers import write_campaign_yaml, write_records


@register_plot("test_plot_cli_minimal")
def _plot_minimal(ctx: PlotContext, params: dict) -> None:
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    out = ctx.output_dir / ctx.filename
    out.write_text(f"ok:{params.get('tag', 'none')}")


def test_plot_cli_writes_output(tmp_path):
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    write_records(records)
    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(
        campaign,
        workdir=workdir,
        records_path=records,
        plots=[{"name": "mini", "kind": "test_plot_cli_minimal", "params": {"tag": "demo"}}],
    )

    app = _build()
    runner = CliRunner()
    res = runner.invoke(app, ["--no-color", "plot", "-c", str(campaign)])
    assert res.exit_code == 0, res.stdout

    out_path = Path(workdir) / "outputs" / "plots" / "mini.png"
    assert out_path.exists()


def test_plot_cli_accepts_directory(tmp_path):
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    write_records(records)
    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(
        campaign,
        workdir=workdir,
        records_path=records,
        plots=[{"name": "mini", "kind": "test_plot_cli_minimal", "params": {"tag": "demo"}}],
    )

    app = _build()
    runner = CliRunner()
    res = runner.invoke(app, ["--no-color", "plot", "-c", str(workdir)])
    assert res.exit_code == 0, res.stdout


def test_plot_cli_rejects_top_level_plot_keys(tmp_path):
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    write_records(records)
    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(
        campaign,
        workdir=workdir,
        records_path=records,
        plots=[
            {
                "name": "mini",
                "kind": "test_plot_cli_minimal",
                "params": {"tag": "demo"},
                "hue": "round",  # invalid top-level plot key
            }
        ],
    )

    app = _build()
    runner = CliRunner()
    res = runner.invoke(app, ["--no-color", "plot", "-c", str(campaign)])
    assert res.exit_code == 1, res.stdout
