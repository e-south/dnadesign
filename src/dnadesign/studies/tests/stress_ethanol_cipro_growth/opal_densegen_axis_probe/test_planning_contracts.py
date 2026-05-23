from __future__ import annotations

from .helpers import (
    AXIS_CLASS_TO_LOGIC4,
    Namespace,
    Path,
    ProbeArtifactLayout,
    _compact_split_metadata,
    _persisted_split_metadata,
    _run_command,
    _split_metadata_for_all,
    build_plan,
    json,
    pd,
    prepare_probe_run_root,
    pytest,
    sys,
    validate_run_root_policy,
    validate_scratch_paths,
)


def test_persisted_split_metadata_keeps_large_id_lists_out_of_json() -> None:
    metadata = {
        "random_id": {
            "split_id": "random_id",
            "budget": 96,
            "per_class": 24,
            "seed": 7,
            "train_ids": ["train-1", "train-2"],
            "eval_ids": ["eval-1", "eval-2", "eval-3"],
        }
    }

    compact = _compact_split_metadata(metadata)
    persisted = _persisted_split_metadata(metadata)

    assert compact["random_id"]["train_count"] == 2
    assert compact["random_id"]["eval_count"] == 3
    assert "train_ids" not in persisted["random_id"]
    assert "eval_ids" not in persisted["random_id"]
    assert persisted["random_id"]["train_ids_path"] == "random_id_train_ids.parquet"
    assert persisted["random_id"]["eval_ids_path"] == "random_id_eval_ids.parquet"


def test_validate_scratch_paths_rejects_shared_observed_label_sidecar(tmp_path: Path) -> None:
    shared = tmp_path / "src/dnadesign/usr/datasets/usr_prom_eth_cip_opal_candidates/_opal/observed_labels.parquet"
    run_root = tmp_path / ".var/studies/stress_ethanol_cipro_growth/opal_densegen_axis_probe/run"

    with pytest.raises(ValueError, match="shared observed-label"):
        validate_scratch_paths(run_root=run_root, label_sidecar_path=shared)


def test_validate_run_root_policy_rejects_apply_writes_outside_study_var(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    with pytest.raises(ValueError, match="run root must be under"):
        validate_run_root_policy(repo_root=repo_root, run_root=tmp_path / "outside")


def test_validate_run_root_policy_allows_default_study_var(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    run_root = repo_root / ".var/studies/stress_ethanol_cipro_growth/opal_densegen_axis_probe/run-1"

    validate_run_root_policy(repo_root=repo_root, run_root=run_root)


def test_validate_run_root_policy_rejects_custom_repo_local_writes(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    run_root = repo_root / "docs/studies/stress_ethanol_cipro_growth/contexts/opal/generated-run"

    with pytest.raises(ValueError, match="custom run root inside the repository"):
        validate_run_root_policy(repo_root=repo_root, run_root=run_root, allow_custom=True)


def test_validate_run_root_policy_allows_explicit_external_scratch(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    run_root = tmp_path / "external-scratch" / "probe-run"

    validate_run_root_policy(repo_root=repo_root, run_root=run_root, allow_custom=True)


def test_build_plan_dry_run_does_not_require_apply_for_source_gate(tmp_path: Path) -> None:
    plan = build_plan(run_root=tmp_path / "probe", initial_label_count=6, seed=7, gate="source", splits=("random_id",))

    assert plan.apply is False
    assert plan.runs == []
    assert plan.commands == []


def test_probe_dry_run_json_reports_planned_plan_path_without_artifact_claim(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dnadesign.studies.studies.stress_ethanol_cipro_growth.opal_densegen_axis_probe import run_cli

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    run_root = repo_root / ".var/studies/stress_ethanol_cipro_growth/opal_densegen_axis_probe/dry"
    labels = pd.DataFrame(
        [
            {"id": "a", "axis_class": "background_only", "quality_flag": "ok", "sigma35_variant": "f"},
            {"id": "b", "axis_class": "cipro_only", "quality_flag": "ok", "sigma35_variant": "f"},
        ]
    )
    monkeypatch.setattr(run_cli, "_repo_root_from", lambda _cwd: repo_root)
    monkeypatch.setattr(run_cli, "_load_candidate_inputs", lambda _repo: (pd.DataFrame(), pd.DataFrame()))
    monkeypatch.setattr(run_cli, "build_axis_oracle", lambda *_args, **_kwargs: labels)
    monkeypatch.setattr(
        run_cli,
        "validate_candidate_x_surface",
        lambda *_args, **_kwargs: {"row_count": len(labels), "x_dim": 8192},
    )
    args = Namespace(
        run_id="dry",
        run_root=run_root,
        apply=False,
        allow_custom_run_root=False,
        initial_labels=6,
        selection_k=6,
        max_x_matrix_gib=None,
        score_batch_size=None,
        seed=7,
        rounds=1,
        gate="source",
        splits="random_id",
        stop_after="status",
        json=True,
        replace_run_root=False,
    )

    assert run_cli._run_probe(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"].endswith(".plan_preview.v1")
    assert payload["mode"] == "dry_run"
    assert payload["writes_artifacts"] is False
    assert "plan_path" not in payload
    assert payload["planned_plan_path"].endswith("probe_plan.json")
    assert payload["plan_exists"] is False
    assert payload["run_root_exists"] is False
    assert not run_root.exists()


def test_build_plan_stop_after_validate_avoids_scoring_commands(tmp_path: Path) -> None:
    plan = build_plan(
        run_root=tmp_path / "probe",
        initial_label_count=6,
        seed=7,
        gate="cipro-random",
        splits=("random_id",),
        stop_after="validate",
    )

    rendered = [" ".join(command) for command in plan.commands]
    assert len(rendered) == 2
    assert all("opal validate" in command for command in rendered)
    assert all("opal run" not in command for command in rendered)


def test_build_plan_multi_round_commands_include_followup_ingest_and_run(tmp_path: Path) -> None:
    plan = build_plan(
        run_root=tmp_path / "probe",
        initial_label_count=6,
        seed=7,
        gate="cipro-random",
        splits=("random_id",),
        rounds=3,
        stop_after="status",
    )

    rendered = [" ".join(command) for command in plan.commands]
    assert plan.rounds == 3
    assert len(rendered) == 18
    assert sum("opal ingest-y" in command for command in rendered) == 6
    assert sum("opal run" in command for command in rendered) == 6
    assert any("--round 2" in command and "vec8-b2.parquet" in command for command in rendered)


def test_build_plan_separates_initial_labels_from_selection_k(tmp_path: Path) -> None:
    plan = build_plan(
        run_root=tmp_path / "probe",
        initial_label_count=8,
        selection_k=6,
        seed=7,
        gate="cipro-random",
        splits=("random_id",),
        rounds=2,
    )

    assert plan.initial_label_count == 8
    assert plan.selection_k == 6
    assert {run.selection_k for run in plan.runs} == {6}


def test_build_plan_rejects_unknown_stop_stage(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported stop_after"):
        build_plan(
            run_root=tmp_path / "probe",
            initial_label_count=6,
            seed=7,
            gate="cipro-random",
            splits=("random_id",),
            stop_after="score-everything-now",
        )


def test_build_plan_rejects_invalid_round_count(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="rounds"):
        build_plan(
            run_root=tmp_path / "probe",
            initial_label_count=6,
            seed=7,
            gate="cipro-random",
            splits=("random_id",),
            rounds=0,
        )


def test_split_metadata_keeps_full_eval_pool_for_realistic_probe(tmp_path: Path) -> None:
    rows = []
    for axis_class in AXIS_CLASS_TO_LOGIC4:
        for idx in range(8):
            rows.append(
                {
                    "id": f"{axis_class}-{idx}",
                    "axis_class": axis_class,
                    "quality_flag": "ok",
                    "sigma35_variant": "f",
                }
            )
    labels = pd.DataFrame(rows)
    plan = build_plan(
        run_root=tmp_path / "probe",
        initial_label_count=6,
        selection_k=6,
        seed=7,
        gate="cipro-random",
        splits=("random_id",),
    )

    metadata = _split_metadata_for_all(labels, plan=plan)["random_id"]

    assert len(metadata["train_ids"]) == 6
    assert len(metadata["eval_ids"]) == 26
    assert "candidate_cap_per_split" not in metadata
    assert "eval_full_count" not in metadata


def test_probe_plan_fingerprint_refuses_mismatched_existing_run_root(tmp_path: Path) -> None:
    layout = ProbeArtifactLayout(tmp_path / "probe")
    first = {"run_root": str(layout.run_root), "rounds": 1, "gate": "source"}
    second = {"run_root": str(layout.run_root), "rounds": 2, "gate": "source"}

    first_record = prepare_probe_run_root(layout, plan_payload=first)

    with pytest.raises(RuntimeError, match="probe plan fingerprint mismatch"):
        prepare_probe_run_root(layout, plan_payload=second)
    assert json.loads(layout.probe_plan_path.read_text(encoding="utf-8"))["fingerprint"] == first_record["fingerprint"]


def test_probe_plan_fingerprint_replaces_legacy_run_root_only_when_explicit(tmp_path: Path) -> None:
    layout = ProbeArtifactLayout(tmp_path / "probe")
    layout.run_root.mkdir(parents=True)
    stale = layout.run_root / "stale.txt"
    stale.write_text("old\n", encoding="utf-8")
    plan_payload = {"run_root": str(layout.run_root), "rounds": 1, "gate": "source"}

    with pytest.raises(RuntimeError, match="artifacts but no probe_plan"):
        prepare_probe_run_root(layout, plan_payload=plan_payload)

    record = prepare_probe_run_root(layout, plan_payload=plan_payload, replace_run_root=True)

    assert not stale.exists()
    assert json.loads(layout.probe_plan_path.read_text(encoding="utf-8"))["fingerprint"] == record["fingerprint"]


def test_run_command_keeps_child_stdout_off_machine_readable_stdout(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _run_command(
        [sys.executable, "-c", "print('child-stdout')"],
        cwd=tmp_path,
        machine_readable=True,
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "child-stdout" in captured.err


def test_source_gate_does_not_report_unused_split_metadata(tmp_path: Path) -> None:
    labels = pd.DataFrame(
        [
            {"id": axis_class, "axis_class": axis_class, "quality_flag": "ok", "sigma35_variant": "f"}
            for axis_class in AXIS_CLASS_TO_LOGIC4
        ]
    )
    plan = build_plan(
        run_root=tmp_path / "probe",
        initial_label_count=4,
        seed=7,
        gate="source",
        splits=("random_id", "leave_sigma35_variant"),
    )

    assert _split_metadata_for_all(labels, plan=plan) == {}
