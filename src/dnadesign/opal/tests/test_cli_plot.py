"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_cli_plot.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from pathlib import Path

from typer.testing import CliRunner

from dnadesign.opal.src.cli.app import _build
from dnadesign.opal.src.plots._context import PlotContext
from dnadesign.opal.src.registries.plots import register_plot

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


def test_plot_cli_list_registry(tmp_path):
    app = _build()
    runner = CliRunner()
    res = runner.invoke(app, ["--no-color", "plot", "--list"])
    assert res.exit_code == 0, res.stdout
    assert "test_plot_cli_minimal" in res.stdout


def test_plot_cli_list_registry_includes_sfxi_diagnostics(tmp_path):
    app = _build()
    runner = CliRunner()
    res = runner.invoke(app, ["--no-color", "plot", "--list"])
    assert res.exit_code == 0, res.stdout
    for name in [
        "sfxi_factorial_effects",
        "sfxi_setpoint_decomposition",
        "sfxi_setpoint_sweep",
        "sfxi_support_diagnostics",
        "sfxi_uncertainty",
        "sfxi_intensity_scaling",
    ]:
        assert name in res.stdout


def test_plot_cli_list_registry_ignores_config(tmp_path):
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    campaign = workdir / "campaign.yaml"
    campaign.write_text("[]\n")  # invalid campaign yaml (not a mapping)

    app = _build()
    runner = CliRunner()
    res = runner.invoke(app, ["--no-color", "plot", "--list", "-c", str(campaign)])
    assert res.exit_code == 0, res.stdout
    assert "Registered plots" in res.stdout


def test_plot_cli_describe(tmp_path):
    app = _build()
    runner = CliRunner()
    res = runner.invoke(app, ["--no-color", "plot", "--describe", "scatter_score_vs_rank"])
    assert res.exit_code == 0, res.stdout
    assert "scatter_score_vs_rank" in res.stdout


def test_plot_cli_list_configured(tmp_path):
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
    res = runner.invoke(app, ["--no-color", "plot", "--list-config", "-c", str(campaign)])
    assert res.exit_code == 0, res.stdout
    assert "mini: test_plot_cli_minimal" in res.stdout


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


def test_plot_cli_rejects_run_id_round_mismatch(tmp_path):
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
    from ._cli_helpers import write_ledger

    write_ledger(workdir, run_id="r0", round_index=0)
    write_ledger(workdir, run_id="r1", round_index=1)
    app = _build()
    runner = CliRunner()
    res = runner.invoke(
        app,
        ["--no-color", "plot", "-c", str(campaign), "--round", "1", "--run-id", "r0"],
    )
    assert res.exit_code != 0, res.stdout
    assert "run_id" in res.output


def test_plot_cli_rejects_bad_round_selector(tmp_path):
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
    from ._cli_helpers import write_ledger

    write_ledger(workdir, run_id="r0", round_index=0)

    app = _build()
    runner = CliRunner()
    res = runner.invoke(app, ["--no-color", "plot", "-c", str(campaign), "--round", "bad"])
    assert res.exit_code != 0, res.stdout
    assert "Invalid round selector" in res.output


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
