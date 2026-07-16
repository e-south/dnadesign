"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/cli/test_cli_plot.py

Regression tests for CLI plot OPAL CLI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import json
import time
from pathlib import Path

import yaml
from typer.testing import CliRunner

from dnadesign.opal.src.cli.app import _build
from dnadesign.opal.src.core.utils import ExitCodes
from dnadesign.opal.src.plots._context import PlotContext
from dnadesign.opal.src.plots._round_overlay import resolve_highlight_round
from dnadesign.opal.src.plots.config import list_configured_plot_specs, load_plot_config
from dnadesign.opal.src.plots.manifests import write_plot_manifest_index
from dnadesign.opal.src.plots.runner import _merged_manifest_index_rows
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


def test_plot_cli_name_filter_preserves_other_manifest_index_entries(tmp_path):
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
            {"name": "mini_a", "kind": "test_plot_cli_minimal", "params": {"tag": "a"}},
            {"name": "mini_b", "kind": "test_plot_cli_minimal", "params": {"tag": "b"}},
        ],
    )

    app = _build()
    runner = CliRunner()
    res = runner.invoke(app, ["--no-color", "plot", "-c", str(campaign)])
    assert res.exit_code == 0, res.stdout
    time.sleep(0.01)
    records.write_bytes(records.read_bytes())

    res = runner.invoke(app, ["--no-color", "plot", "-c", str(campaign), "--name", "mini_a"])
    assert res.exit_code == 0, res.stdout

    index = json.loads((workdir / "outputs" / "plots" / "plot_manifest.json").read_text())
    assert index["plot_count"] == 2
    assert {row["name"] for row in index["manifests"]} == {"mini_a", "mini_b"}
    freshness_by_name = {row["name"]: row["freshness"]["status"] for row in index["manifests"]}
    assert freshness_by_name == {"mini_a": "fresh", "mini_b": "stale"}
    mini_b_manifest = json.loads((workdir / "outputs" / "plots" / "mini_b.manifest.json").read_text())
    assert mini_b_manifest["freshness"]["status"] == "stale"
    assert (workdir / "outputs" / "plots" / "mini_b.png").exists()


def test_targeted_manifest_merge_preserves_same_name_round_variant_entries(tmp_path):
    output_dir = tmp_path / "plots"
    output_dir.mkdir()
    existing_rows = [
        _manifest_row(output_dir, name="score_by_round", plot_id="score_by_round_r0"),
        _manifest_row(output_dir, name="score_by_round", plot_id="score_by_round_r1"),
        _manifest_row(output_dir, name="other_plot", plot_id="other_plot_rall"),
    ]
    write_plot_manifest_index(output_dir, existing_rows)
    rerun_row = _manifest_row(output_dir, name="score_by_round", plot_id="score_by_round_r2")

    merged = _merged_manifest_index_rows(output_dir, [rerun_row], merge_existing=True)

    assert {row["plot_id"] for row in merged} == {
        "score_by_round_r0",
        "score_by_round_r1",
        "score_by_round_r2",
        "other_plot_rall",
    }
    assert sum(row["name"] == "score_by_round" for row in merged) == 3


def _manifest_row(output_dir: Path, *, name: str, plot_id: str) -> dict:
    media_path = output_dir / f"{plot_id}.png"
    media_path.write_text(plot_id)
    return {
        "schema_version": "opal.plot_artifact.v1",
        "plot_id": plot_id,
        "name": name,
        "kind": "test_plot_cli_minimal",
        "status": "written",
        "manifest_path": str(output_dir / f"{plot_id}.manifest.json"),
        "inputs": [],
        "outputs": [{"role": "media", "path": str(media_path), "exists": True}],
        "freshness": {"schema_version": "opal.plot_freshness.v1", "status": "fresh"},
        "warnings": [],
    }


def test_plot_cli_plot_local_round_selector_overrides_global_round(tmp_path):
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
                "round_selector": "latest",
            }
        ],
    )

    app = _build()
    runner = CliRunner()
    res = runner.invoke(app, ["--no-color", "plot", "-c", str(campaign), "--round", "all"])
    assert res.exit_code == 0, res.stdout

    out_path = Path(workdir) / "outputs" / "plots" / "mini_rlatest.png"
    manifest_path = Path(workdir) / "outputs" / "plots" / "mini_rlatest.manifest.json"
    assert out_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["name"] == "mini"
    assert manifest["rounds"] == "latest"


def test_plot_cli_round_variants_write_manifested_scope_artifacts(tmp_path):
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
                "round_variants": ["all", "each"],
            }
        ],
    )
    from dnadesign.opal.tests._cli_helpers import write_ledger

    write_ledger(workdir, run_id="r0", round_index=0)
    write_ledger(workdir, run_id="r1", round_index=1)

    app = _build()
    runner = CliRunner()
    res = runner.invoke(app, ["--no-color", "plot", "-c", str(campaign), "--round", "latest"])
    assert res.exit_code == 0, res.stdout

    index = json.loads((workdir / "outputs" / "plots" / "plot_manifest.json").read_text())
    assert index["plot_count"] == 3
    manifests = {
        path.name: json.loads(path.read_text())
        for path in sorted((workdir / "outputs" / "plots").glob("mini*.manifest.json"))
    }
    assert manifests["mini_rall.manifest.json"]["rounds"] == "all"
    assert manifests["mini_r0.manifest.json"]["rounds"] == [0]
    assert manifests["mini_r0.manifest.json"]["run_id"] == "r0"
    assert manifests["mini_r1.manifest.json"]["rounds"] == [1]
    assert manifests["mini_r1.manifest.json"]["run_id"] == "r1"


def test_plot_cli_rejects_each_variant_for_inherent_round_history_plots(tmp_path):
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
                "name": "score_history",
                "kind": "metric_over_rounds",
                "round_variants": ["all", "each"],
            }
        ],
    )

    app = _build()
    runner = CliRunner()
    res = runner.invoke(app, ["--no-color", "plot", "-c", str(campaign), "--round", "all"])

    assert res.exit_code != 0
    assert "round_scope=round_history" in str(res.exception)


def test_round_highlight_overlay_resolves_round_zero() -> None:
    assert resolve_highlight_round(0, [0, 1]) == 0
    assert resolve_highlight_round("latest", [0, 1]) == 1
    assert resolve_highlight_round(False, [0, 1]) is None


def test_metric_over_rounds_highlight_does_not_draw_vertical_round_marker(tmp_path, monkeypatch) -> None:
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
                "params": {
                    "metric": "view__selection_score",
                    "cohort": "selected",
                    "summaries": ["median"],
                    "highlight_round": "latest",
                },
            }
        ],
    )
    from dnadesign.opal.tests._cli_helpers import write_ledger, write_state

    write_state(workdir, records_path=records, run_id="r0", round_index=0)
    write_ledger(workdir, run_id="r0", round_index=0)
    write_state(workdir, records_path=records, run_id="r1", round_index=1)
    write_ledger(workdir, run_id="r1", round_index=1)

    import matplotlib.axes

    def fail_axvline(self, *args, **kwargs):
        raise AssertionError("metric_over_rounds should not draw a vertical round marker")

    monkeypatch.setattr(matplotlib.axes.Axes, "axvline", fail_axvline)

    app = _build()
    runner = CliRunner()
    res = runner.invoke(app, ["--no-color", "plot", "-c", str(campaign), "--round", "all"])

    assert res.exit_code == 0, res.stdout


def test_metric_over_rounds_defaults_to_mean_only_and_preserves_metric_expression(tmp_path) -> None:
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
                "params": {
                    "metric": "view__selection_score",
                    "cohort": "selected",
                    "metric_label": "Score = -MSE(y_hat, [0, 0, 1, 1])",
                    "legend_metric_label": "negative MSE score",
                    "metric_expression": (
                        "score = -mean((y_hat - [0, 0, 1, 1])^2); loss = mean((y_hat - [0, 0, 1, 1])^2)"
                    ),
                },
                "output": {"save_data": True},
            }
        ],
    )
    from dnadesign.opal.tests._cli_helpers import write_ledger, write_state

    write_state(workdir, records_path=records, run_id="r0", round_index=0)
    write_ledger(workdir, run_id="r0", round_index=0)

    app = _build()
    runner = CliRunner()
    res = runner.invoke(app, ["--no-color", "plot", "-c", str(campaign), "--round", "all"])

    assert res.exit_code == 0, res.stdout
    import pandas as pd

    tidy = pd.read_csv(workdir / "outputs" / "plots" / "metric_rall.csv")
    assert sorted(tidy["summary"].unique().tolist()) == ["mean"]
    manifest = json.loads((workdir / "outputs" / "plots" / "metric_rall.manifest.json").read_text())
    assert manifest["params"]["metric_label"] == "Score = -MSE(y_hat, [0, 0, 1, 1])"
    assert "loss = mean" in manifest["params"]["metric_expression"]


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


def test_plot_cli_list_configured_json_error_when_config_missing():
    app = _build()
    runner = CliRunner()
    res = runner.invoke(app, ["--no-color", "plot", "--list-config", "--json"])

    assert res.exit_code == ExitCodes.BAD_ARGS, res.stdout
    payload = json.loads(res.stdout)
    assert payload["ok"] is False
    assert payload["error"]["schema_version"] == "opal.cli_error.v1"
    assert payload["error"]["context"] == "plot list-config"
    assert "No config provided" in payload["error"]["message"]
    assert "Traceback" not in res.stdout
    assert "Traceback" not in res.stderr


def test_stress_rmf_campaign_declares_concise_plot_policy() -> None:
    config_path = Path("src/dnadesign/opal/campaigns/secg_rmf_greedy/configs/campaign.yaml")
    expected = {
        "rmf_candidate_frontier": "response_magnitude_feasibility_frontier",
        "rmf_score_vs_rank": "scatter_score_vs_rank",
        "rmf_selected_constraints": "response_magnitude_feasibility_constraint_decomposition",
    }

    campaign_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    plot_cfg = load_plot_config(
        campaign_cfg=campaign_cfg,
        campaign_yaml=config_path,
        plot_config_opt=None,
    )
    specs = list_configured_plot_specs(
        plots_cfg=plot_cfg.plots,
        plot_presets=plot_cfg.plot_presets,
    )
    assert {spec["name"]: spec["kind"] for spec in specs} == expected
    assert all(spec["round_selector"] == "latest" for spec in specs)
    assert all(spec.get("round_variants") is None for spec in specs)


def test_plot_cli_accepts_directory(tmp_path):
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    write_records(records)
    campaign = workdir / "configs" / "campaign.yaml"
    campaign.parent.mkdir(parents=True)
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
    assert "Traceback" not in res.output


def test_plot_cli_missing_runs_ledger_returns_json_error_without_traceback(tmp_path):
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
    res = runner.invoke(
        app,
        ["--no-color", "plot", "-c", str(campaign), "--run-id", "missing", "--json", "--name", "mini"],
    )

    assert res.exit_code != 0, res.stdout
    payload = json.loads(res.output)
    assert payload["error"]["schema_version"] == "opal.cli_error.v1"
    assert payload["error"]["context"] == "plot"
    assert "Missing runs sink" in payload["error"]["message"]
    assert "Traceback" not in res.output


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
    assert "Traceback" not in res.output


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
                "params": {
                    "metric": "view__selection_score",
                    "cohort": "selected",
                    "summaries": ["median", "q25", "q75"],
                    "band": "iqr",
                    "highlight_round": "latest",
                },
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
                "params": {
                    "cohort": "all_pool",
                    "channel_labels": ["y0"],
                    "include_reference_vector": True,
                    "reference_vector": [0.0],
                },
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
        if name == "feature_heat":
            assert width > height
        else:
            assert width == height


def test_feature_importance_heatmap_rejects_sort_alias(tmp_path):
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
                "name": "feature_heat",
                "kind": "feature_importance_heatmap",
                "params": {"sort": "sort_index"},
            }
        ],
    )

    app = _build()
    runner = CliRunner()
    res = runner.invoke(app, ["--no-color", "plot", "-c", str(campaign)])

    assert res.exit_code == 1, res.stdout
    manifests = sorted((workdir / "outputs" / "plots").glob("feature_heat*.manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text())
    assert manifest["status"] == "failed"
    assert "does not accept parameter 'sort'" in manifest["error"]["message"]


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
