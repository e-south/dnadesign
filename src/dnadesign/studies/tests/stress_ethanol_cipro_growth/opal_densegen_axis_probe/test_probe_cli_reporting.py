from __future__ import annotations

from .helpers import Path, json, pytest, summarize_probe_progress


def test_probe_progress_summarizes_round_logs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from dnadesign.opal.tests._cli_helpers import write_campaign_yaml, write_records, write_state
    from dnadesign.studies.studies.stress_ethanol_cipro_growth.opal_densegen_axis_probe.cli import main as probe_main

    run_root = tmp_path / "probe"
    workdir = run_root / "scratch_campaigns" / "cipro_positive_random_id"
    records = workdir / "records.parquet"
    records.parent.mkdir(parents=True)
    write_records(records)
    config_path = workdir / "configs" / "campaign.yaml"
    config_path.parent.mkdir(parents=True)
    write_campaign_yaml(config_path, workdir=workdir, records_path=records)
    write_state(workdir, records_path=records, run_id="run-0", round_index=0)
    log_path = (
        run_root
        / "scratch_campaigns"
        / "cipro_positive_random_id"
        / "outputs"
        / "rounds"
        / "round_0"
        / "logs"
        / "round.log.jsonl"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "\n".join(
            [
                json.dumps({"ts": "2026-05-19T18:36:23+00:00", "stage": "start"}),
                json.dumps({"ts": "2026-05-19T18:36:30+00:00", "stage": "predict_batch", "batch": 2, "of": 5}),
                json.dumps({"ts": "2026-05-19T18:36:35+00:00", "stage": "done"}),
            ]
        ),
        encoding="utf-8",
    )

    payload = summarize_probe_progress(run_root)

    assert payload["status"] == "done"
    assert payload["detail"] == "compact"
    assert payload["campaign_count"] == 1
    campaign = payload["campaigns"][0]
    assert campaign["run_key"] == "cipro_positive_random_id"
    assert campaign["round_index"] == 0
    assert campaign["last_stage"] == "done"
    assert campaign["predict"]["batch"] == 2
    assert "opal_progress" not in campaign
    assert probe_main(["progress", "--run-root", str(run_root), "--json"]) == 0
    progress_json = json.loads(capsys.readouterr().out)
    assert progress_json["schema_version"].endswith(".progress.v1")
    assert progress_json["detail"] == "compact"
    assert "opal_progress" not in progress_json["campaigns"][0]

    assert probe_main(["progress", "--run-root", str(run_root), "--json", "--full"]) == 0
    full_progress_json = json.loads(capsys.readouterr().out)
    assert full_progress_json["detail"] == "full"
    assert "opal_progress" in full_progress_json["campaigns"][0]


def test_probe_progress_uses_planned_round_count_for_done_status(tmp_path: Path) -> None:
    from dnadesign.opal.tests._cli_helpers import write_campaign_yaml, write_records, write_state

    run_root = tmp_path / "probe"
    workdir = run_root / "scratch_campaigns" / "cipro_positive_random_id"
    records = workdir / "records.parquet"
    records.parent.mkdir(parents=True)
    write_records(records)
    config_path = workdir / "configs" / "campaign.yaml"
    config_path.parent.mkdir(parents=True)
    write_campaign_yaml(config_path, workdir=workdir, records_path=records)
    write_state(workdir, records_path=records, run_id="run-0", round_index=0)
    log_path = workdir / "outputs" / "rounds" / "round_0" / "logs" / "round.log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "\n".join(
            [
                json.dumps({"ts": "2026-05-19T18:36:23+00:00", "stage": "start"}),
                json.dumps({"ts": "2026-05-19T18:36:35+00:00", "stage": "done"}),
            ]
        ),
        encoding="utf-8",
    )
    (run_root / "probe_plan.json").write_text(
        json.dumps({"plan": {"planned_runs": 1, "rounds": 12, "gate": "cipro-random", "stop_after": "status"}}),
        encoding="utf-8",
    )

    payload = summarize_probe_progress(run_root)

    assert payload["status"] == "running_or_incomplete"
    assert payload["expected_round_count"] == 12
    assert payload["campaigns"][0]["status"] == "running_or_incomplete"
    assert payload["campaigns"][0]["round_count"] == 1


def test_probe_plot_json_redirects_noisy_generation_stdout(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dnadesign.studies.studies.stress_ethanol_cipro_growth.opal_densegen_axis_probe import plotting
    from dnadesign.studies.studies.stress_ethanol_cipro_growth.opal_densegen_axis_probe.cli import main as probe_main

    def noisy_plot(run_root: Path, *, round_selector: str, quiet: bool = False) -> dict[str, object]:
        print("accidental plot stdout")
        return {
            "schema_version": "stress_ethanol_cipro_growth.opal_densegen_axis_probe.plot.v1",
            "run_root": str(run_root),
            "round_selector": round_selector,
            "quiet": quiet,
            "any_fail": False,
        }

    monkeypatch.setattr(plotting, "generate_probe_campaign_plots", noisy_plot)

    assert probe_main(["plot", "--run-root", str(tmp_path / "probe"), "--json"]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["quiet"] is True
    assert "accidental plot stdout" not in captured.out
    assert "accidental plot stdout" in captured.err


def test_probe_report_json_redirects_noisy_review_stdout(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dnadesign.studies.studies.stress_ethanol_cipro_growth.opal_densegen_axis_probe import review
    from dnadesign.studies.studies.stress_ethanol_cipro_growth.opal_densegen_axis_probe.cli import main as probe_main

    def noisy_review(run_root: Path, *, include_plots: bool = True) -> dict[str, object]:
        print("accidental report stdout")
        return {
            "schema_version": "stress_ethanol_cipro_growth.opal_densegen_axis_probe.review.v1",
            "run_root": str(run_root),
            "plots": include_plots,
            "decision": "DEBUG",
            "status": "attention",
        }

    monkeypatch.setattr(review, "build_probe_review", noisy_review)

    assert probe_main(["report", "--run-root", str(tmp_path / "probe"), "--json"]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["decision"] == "DEBUG"
    assert "accidental report stdout" not in captured.out
    assert "accidental report stdout" in captured.err
