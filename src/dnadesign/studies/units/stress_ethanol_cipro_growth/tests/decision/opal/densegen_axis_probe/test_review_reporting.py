"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/densegen_axis_probe/test_review_reporting.py

Regression tests for review reporting studies units stress ethanol cipro growth.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .helpers import (
    ORACLE_ID,
    Path,
    _valid_metrics_payload,
    build_probe_review,
    json,
    pd,
    pytest,
)
from .probe_modules import probe_module


def test_probe_report_reuses_opal_campaign_review_primitives(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from dnadesign.opal.tests._cli_helpers import write_campaign_yaml, write_ledger, write_records, write_state

    probe_main = probe_module("cli").main

    run_root = tmp_path / "probe"
    reports_dir = run_root / "reports"
    reports_dir.mkdir(parents=True)
    workdir = run_root / "scratch_campaigns" / "cipro_positive_random_id"
    records = workdir / "records.parquet"
    records.parent.mkdir(parents=True)
    write_records(records)
    config_path = workdir / "configs" / "campaign.yaml"
    config_path.parent.mkdir(parents=True)
    write_campaign_yaml(config_path, workdir=workdir, records_path=records)
    write_state(workdir, records_path=records, run_id="run-0", round_index=0)
    write_ledger(workdir, run_id="run-0", round_index=0)
    feature_dir = workdir / "outputs" / "rounds" / "round_0" / "model"
    feature_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"feature_index": [0, 1], "importance": [0.2, 0.8]}).to_csv(
        feature_dir / "feature_importance.csv",
        index=False,
    )
    plots_dir = workdir / "outputs" / "plots"
    plots_dir.mkdir(parents=True)
    media_path = plots_dir / "score_selected_over_rounds_rall.png"
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (320, 240), "white")
    draw = ImageDraw.Draw(image)
    draw.line((20, 220, 300, 20), fill="black", width=3)
    image.save(media_path)
    csv_path = plots_dir / "score_selected_over_rounds_rall.csv"
    pd.DataFrame(
        {
            "round": [0],
            "cohort": ["selected"],
            "metric": ["view__selection_score"],
            "summary": ["mean"],
            "value": [0.5],
        }
    ).to_csv(csv_path, index=False)
    plot_manifest_path = plots_dir / "score_selected_over_rounds_rall.manifest.json"
    plot_manifest = {
        "schema_version": "opal.plot_artifact.v1",
        "name": "score_selected_over_rounds",
        "kind": "metric_over_rounds",
        "status": "written",
        "generated_at": "2026-05-20T00:00:00+00:00",
        "run_id": "run-0",
        "rounds": "all",
        "params": {},
        "outputs": [
            {"role": "media", "path": str(media_path), "exists": True},
            {"role": "tidy_csv", "path": str(csv_path), "exists": True},
        ],
        "manifest_path": str(plot_manifest_path),
    }
    plot_manifest_path.write_text(json.dumps(plot_manifest), encoding="utf-8")
    (plots_dir / "plot_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "opal.plot_manifest_index.v1",
                "generated_at": "2026-05-20T00:00:00+00:00",
                "output_dir": str(plots_dir),
                "plot_count": 1,
                "manifests": [plot_manifest],
            }
        ),
        encoding="utf-8",
    )
    metric_row = {
        "run_key": "cipro_positive_random_id",
        "campaign": "cipro",
        "oracle_id": ORACLE_ID,
        "split_id": "random_id",
        "target_class": "cipro_only",
        "train_count": 1,
        "eval_count": 2,
        "run_id": "run-0",
        "as_of_round": 0,
        "selection_k": 6,
        "selected_count_in_eval": 6,
        "selected_target_precision_at_k_true": 0.5,
        "target_lift_at_k_true": 2.0,
        "off_target_class_distribution_true": {
            "background_only": 0,
            "ethanol_only": 0,
            "cipro_only": 1,
            "dual_axis_and": 0,
        },
    }
    metrics_payload = _valid_metrics_payload([metric_row])
    metrics_payload["rounds"] = [
        {
            **metric_row,
            "metric_scope": "round",
            "as_of_round": 0,
            "selected_target_count_true": 3,
            "target_class_prevalence_true": 0.25,
        }
    ]
    (reports_dir / "metrics.json").write_text(json.dumps(metrics_payload), encoding="utf-8")
    (reports_dir / "decision.md").write_text(
        "# opal_densegen_axis_probe_v0 decision\n\n## Decision\n\nPASS_CIPRO_RANDOM_GATE\n",
        encoding="utf-8",
    )

    payload = build_probe_review(run_root)

    assert Path(payload["review"]).exists()
    assert Path(payload["index"]).exists()
    assert Path(payload["run_manifest"]).exists()
    opal_review = workdir / "outputs" / "review" / "selection_views" / "primary" / "review.md"
    assert opal_review.exists()
    opal_index = workdir / "outputs" / "review" / "selection_views" / "primary" / "index.html"
    assert opal_index.exists()
    review_text = Path(payload["review"]).read_text(encoding="utf-8")
    assert "OPAL campaign run review artifacts remain campaign-scoped" in review_text
    assert "PASS_CIPRO_RANDOM_GATE" in review_text
    assert "Configured OPAL Plots" in review_text
    assert "Probe Outcome" in review_text
    assert "pre-assay synthetic-oracle learnability probe" in review_text
    index_text = Path(payload["index"]).read_text(encoding="utf-8")
    assert "DenseGen axis probe review" in index_text
    assert "Probe Outcome" in index_text
    assert "cipro_positive_random_id" in index_text
    assert "score_selected_over_rounds" in index_text
    assert "Decision Reasons" in index_text
    assert "Selected Target" in index_text
    assert "Metric Guide" in index_text
    assert "lift = precision@K / target prevalence" in index_text
    manifest = json.loads(Path(payload["review_manifest"]).read_text(encoding="utf-8"))
    assert any(path.endswith("round_target_lift_and_precision.png") for path in manifest["probe_plots"])
    assert (run_root / "reports" / "round_metrics.jsonl").exists()
    assert "metric_definitions" in manifest
    assert manifest["opal_configured_plots"][0]["plot_count"] == 1
    assert manifest["plot_quality"]["plot_count"] == 1
    assert manifest["plot_quality"]["problem_count"] == 0
    assert manifest["outcome_summary"]["decision"] == "DEBUG"
    assert "wet-lab phenotype" in manifest["outcome_summary"]["interpretation_boundary"]
    assert manifest["decision_reasons"]
    assert manifest["gate_results"]
    assert manifest["round_dynamics"][0]["run_key"] == "cipro_positive_random_id"
    status_payload = json.loads((run_root / "reports" / "status.json").read_text(encoding="utf-8"))
    assert status_payload["decision_reasons"]
    assert status_payload["round_dynamics"]
    assert status_payload["outcome_summary"]["next_action"]
    assert probe_main(["report", "--run-root", str(run_root), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["decision"] == "DEBUG"


def test_probe_report_points_to_configured_plot_refresh_when_indexes_missing(tmp_path: Path) -> None:
    from dnadesign.opal.tests._cli_helpers import write_campaign_yaml, write_ledger, write_records, write_state

    run_root = tmp_path / "probe"
    reports_dir = run_root / "reports"
    reports_dir.mkdir(parents=True)
    workdir = run_root / "scratch_campaigns" / "cipro_positive_random_id"
    records = workdir / "records.parquet"
    records.parent.mkdir(parents=True)
    write_records(records)
    config_path = workdir / "configs" / "campaign.yaml"
    config_path.parent.mkdir(parents=True)
    write_campaign_yaml(config_path, workdir=workdir, records_path=records)
    write_state(workdir, records_path=records, run_id="run-0", round_index=0)
    write_ledger(workdir, run_id="run-0", round_index=0)
    metric_row = {
        "run_key": "cipro_positive_random_id",
        "campaign": "cipro",
        "oracle_id": ORACLE_ID,
        "split_id": "random_id",
        "target_class": "cipro_only",
        "train_count": 1,
        "eval_count": 2,
        "run_id": "run-0",
        "as_of_round": 0,
        "selection_k": 6,
        "selected_count_in_eval": 6,
        "selected_target_precision_at_k_true": 0.5,
        "target_lift_at_k_true": 2.0,
        "off_target_class_distribution_true": {
            "background_only": 0,
            "ethanol_only": 0,
            "cipro_only": 1,
            "dual_axis_and": 0,
        },
    }
    (reports_dir / "metrics.json").write_text(json.dumps(_valid_metrics_payload([metric_row])), encoding="utf-8")
    (reports_dir / "decision.md").write_text(
        "# opal_densegen_axis_probe_v0 decision\n\n## Decision\n\nPASS_CIPRO_RANDOM_GATE\n",
        encoding="utf-8",
    )

    payload = build_probe_review(run_root, include_plots=False)

    assert payload["plot_quality"]["status"] == "attention"
    assert payload["plot_quality"]["problem_count"] == 1
    assert "configured_plot_refresh_command" in payload["next_steps"]
    assert " plot --run-root " in payload["next_steps"]["configured_plot_refresh_command"]
    review_text = Path(payload["review"]).read_text(encoding="utf-8")
    assert "configured plot refresh" in review_text


def test_probe_report_recomputes_stale_persisted_decisions(tmp_path: Path) -> None:
    run_root = tmp_path / "probe"
    reports_dir = run_root / "reports"
    reports_dir.mkdir(parents=True)
    payload = _valid_metrics_payload()
    payload["safety"]["x_surface_pass"] = False
    (reports_dir / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")
    (reports_dir / "decision.md").write_text(
        "# opal_densegen_axis_probe_v0 decision\n\n## Decision\n\nPENDING\n",
        encoding="utf-8",
    )

    review = build_probe_review(run_root, include_plots=False)

    assert review["decision"] == "STOP"
    assert review["persisted_decision"] == "PENDING"
    assert review["status"] == "attention"
    manifest = json.loads(Path(review["review_manifest"]).read_text(encoding="utf-8"))
    assert "persisted_decision_mismatch:PENDING!=STOP" in manifest["problems"]
