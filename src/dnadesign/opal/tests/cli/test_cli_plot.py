"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/cli/test_cli_plot.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import json
from pathlib import Path

from typer.testing import CliRunner

from dnadesign.opal.src.cli.app import _build
from dnadesign.opal.src.plots._context import PlotContext
from dnadesign.opal.src.registries.plots import PlotMeta, describe_plot_kind, list_plots, register_plot
from dnadesign.opal.tests._cli_helpers import write_campaign_yaml, write_records


def _png_dimensions(path: Path) -> tuple[int, int]:
    header = path.read_bytes()[:24]
    if not header.startswith(b"\x89PNG\r\n\x1a\n"):
        raise AssertionError(f"not a PNG file: {path}")
    return int.from_bytes(header[16:20], "big"), int.from_bytes(header[20:24], "big")


@register_plot("test_plot_cli_minimal")
def _plot_minimal(ctx: PlotContext, params: dict) -> None:
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    out = ctx.output_dir / ctx.filename
    out.write_text(f"ok:{params.get('tag', 'none')}")


@register_plot("test_plot_cli_no_output")
def _plot_no_output(ctx: PlotContext, params: dict) -> None:
    return None


@register_plot(
    "test_plot_cli_bad_tidy_schema",
    meta=PlotMeta(
        summary="Test plot with intentionally invalid tidy output.",
        data_shape="test table",
        tidy_schema=["a", "b"],
        failure_modes=["missing declared tidy CSV columns"],
    ),
)
def _plot_bad_tidy_schema(ctx: PlotContext, params: dict) -> None:
    import pandas as pd

    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    (ctx.output_dir / ctx.filename).write_text("ok")
    if ctx.save_data:
        ctx.save_df(pd.DataFrame({"a": [1]}))


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
    manifest_path = Path(workdir) / "outputs" / "plots" / "mini.manifest.json"
    index_path = Path(workdir) / "outputs" / "plots" / "plot_manifest.json"
    assert manifest_path.exists()
    assert index_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema_version"] == "opal.plot_artifact.v1"
    assert manifest["name"] == "mini"
    assert manifest["status"] == "written"
    assert manifest["outputs"][0]["role"] == "media"
    index = json.loads(index_path.read_text())
    assert index["schema_version"] == "opal.plot_manifest_index.v1"
    assert index["plot_count"] == 1


def test_plot_cli_list_registry(tmp_path):
    app = _build()
    runner = CliRunner()
    res = runner.invoke(app, ["--no-color", "plot", "--list"])
    assert res.exit_code == 0, res.stdout
    assert "test_plot_cli_minimal" in res.stdout

    res_json = runner.invoke(app, ["--no-color", "plot", "--list", "--json"])
    assert res_json.exit_code == 0, res_json.stdout
    payload = json.loads(res_json.stdout)
    assert payload["schema_version"] == "opal.plot_registry.v1"
    assert any(row["kind"] == "test_plot_cli_minimal" for row in payload["plots"])


def test_plot_cli_list_registry_includes_sfxi_diagnostics(tmp_path):
    app = _build()
    runner = CliRunner()
    res = runner.invoke(app, ["--no-color", "plot", "--list"])
    assert res.exit_code == 0, res.stdout
    for name in [
        "feature_importance_heatmap",
        "metric_over_rounds",
        "sfxi_factorial_effects",
        "sfxi_setpoint_sweep",
        "sfxi_support_diagnostics",
        "sfxi_uncertainty",
        "sfxi_intensity_scaling",
        "vector_summary_heatmap",
    ]:
        assert name in res.stdout
    assert "sfxi_setpoint_decomposition" not in res.stdout


def test_builtin_plot_metadata_declares_shape_and_failure_modes() -> None:
    missing = []
    for kind in list_plots():
        if kind.startswith("test_plot_cli_"):
            continue
        meta = describe_plot_kind(kind)
        if not meta.get("data_shape") or not meta.get("failure_modes"):
            missing.append(kind)

    assert missing == []


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

    res_json = runner.invoke(app, ["--no-color", "plot", "--describe", "scatter_score_vs_rank", "--json"])
    assert res_json.exit_code == 0, res_json.stdout
    payload = json.loads(res_json.stdout)
    assert payload["schema_version"] == "opal.plot_description.v1"
    assert payload["plot"]["kind"] == "scatter_score_vs_rank"

    missing_json = runner.invoke(app, ["--no-color", "plot", "--describe", "definitely_missing_plot", "--json"])
    assert missing_json.exit_code != 0, missing_json.stdout
    error_payload = json.loads(missing_json.stdout)
    assert error_payload["ok"] is False
    assert error_payload["error"]["schema_version"] == "opal.cli_error.v1"
    assert error_payload["error"]["context"] == "plot describe"


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

    res_json = runner.invoke(app, ["--no-color", "plot", "--list-config", "-c", str(campaign), "--json"])
    assert res_json.exit_code == 0, res_json.stdout
    payload = json.loads(res_json.stdout)
    assert payload["schema_version"] == "opal.plot_config.v1"
    assert payload["plots"] == [
        {
            "name": "mini",
            "kind": "test_plot_cli_minimal",
            "enabled": True,
            "tags": [],
        }
    ]


def test_plot_cli_list_configured_json_error_when_no_plots(tmp_path):
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    write_records(records)
    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(campaign, workdir=workdir, records_path=records)

    app = _build()
    runner = CliRunner()
    res = runner.invoke(app, ["--no-color", "plot", "--list-config", "-c", str(campaign), "--json"])

    assert res.exit_code != 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload["ok"] is False
    assert payload["error"]["schema_version"] == "opal.cli_error.v1"
    assert payload["error"]["context"] == "plot list-config"
    assert "No plots found" in payload["error"]["message"]


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
    from dnadesign.opal.tests._cli_helpers import write_ledger

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
    from dnadesign.opal.tests._cli_helpers import write_ledger

    write_ledger(workdir, run_id="r0", round_index=0)

    app = _build()
    runner = CliRunner()
    res = runner.invoke(app, ["--no-color", "plot", "-c", str(campaign), "--round", "bad"])
    assert res.exit_code != 0, res.stdout
    assert "Invalid round selector" in res.output


def test_plot_cli_writes_failed_manifest_when_plugin_does_not_write_output(tmp_path):
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    write_records(records)
    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(
        campaign,
        workdir=workdir,
        records_path=records,
        plots=[{"name": "nope", "kind": "test_plot_cli_no_output"}],
    )

    app = _build()
    runner = CliRunner()
    res = runner.invoke(app, ["--no-color", "plot", "-c", str(campaign)])

    assert res.exit_code == 1, res.stdout
    manifest = json.loads((workdir / "outputs" / "plots" / "nope.manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["error"]["category"] == "PlotDataContractError"


def test_plot_cli_fails_when_tidy_csv_missing_declared_columns(tmp_path):
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
                "name": "bad_tidy",
                "kind": "test_plot_cli_bad_tidy_schema",
                "output": {"save_data": True},
            }
        ],
    )

    app = _build()
    runner = CliRunner()
    res = runner.invoke(app, ["--no-color", "plot", "-c", str(campaign)])

    assert res.exit_code == 1, res.stdout
    manifest = json.loads((workdir / "outputs" / "plots" / "bad_tidy.manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["quality"]["tidy_schema_valid"] is False
    assert manifest["quality"]["missing_tidy_columns"] == ["b"]
    assert manifest["error"]["category"] == "PlotDataContractError"


def test_plot_cli_generic_primitives_write_manifested_data(tmp_path):
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
                "name": "metric",
                "kind": "metric_over_rounds",
                "params": {"metric": "pred__score_selected", "cohort": ["selected", "all_pool"]},
                "output": {"save_data": True},
            },
            {
                "name": "feature_heat",
                "kind": "feature_importance_heatmap",
                "params": {"order_policy": "sort_index"},
                "output": {"save_data": True},
            },
            {
                "name": "vector_heat",
                "kind": "vector_summary_heatmap",
                "params": {"cohort": "all_pool", "channel_labels": ["y0"], "include_setpoint": True, "setpoint": [0.0]},
                "output": {"save_data": True},
            },
        ],
    )
    from dnadesign.opal.tests._cli_helpers import write_ledger, write_state

    write_state(workdir, records_path=records, run_id="r0", round_index=0)
    write_ledger(workdir, run_id="r0", round_index=0)
    feature_dir = workdir / "outputs" / "rounds" / "round_0" / "model"
    feature_dir.mkdir(parents=True, exist_ok=True)
    (feature_dir / "feature_importance.csv").write_text("feature_index,importance\n0,0.25\n1,0.75\n")

    app = _build()
    runner = CliRunner()
    res = runner.invoke(app, ["--no-color", "plot", "-c", str(campaign), "--round", "0", "--run-id", "r0"])

    assert res.exit_code == 0, res.stdout
    index = json.loads((workdir / "outputs" / "plots" / "plot_manifest.json").read_text())
    assert index["plot_count"] == 3
    for name in ["metric", "feature_heat", "vector_heat"]:
        manifest = json.loads((workdir / "outputs" / "plots" / f"{name}_r0.manifest.json").read_text())
        assert manifest["status"] == "written"
        assert manifest["tidy_csv"].endswith(".csv")
        width, height = _png_dimensions(workdir / "outputs" / "plots" / f"{name}_r0.png")
        assert width == height


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
