from __future__ import annotations

import json

from .helpers import (
    CANDIDATE_RECORDS,
    NULL_ORACLE_ID,
    ORACLE_ID,
    X_COLUMN,
    Path,
    ProbeArtifactLayout,
    ProbePlan,
    RunSpec,
    _make_training_input,
    materialize_probe_inputs,
    pd,
    pytest,
    run_opal_rounds_for_probe,
    selected_ids_from_round,
    write_followup_label_input,
)


def test_materialize_probe_inputs_writes_shared_records_symlink_and_candidate_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_records = tmp_path / CANDIDATE_RECORDS
    source_records.parent.mkdir(parents=True)
    ids = ["random-train", "random-eval", "leave-train", "leave-eval"]
    pd.DataFrame(
        {
            "id": ids,
            "sequence": ["AAAA", "CCCC", "GGGG", "TTTT"],
            X_COLUMN: [[0.0, 1.0]] * len(ids),
        }
    ).to_parquet(source_records, index=False)
    run_root = tmp_path / "probe"
    layout = ProbeArtifactLayout(run_root)
    runs = [
        RunSpec(
            campaign_key="cipro",
            oracle_id=ORACLE_ID,
            split_id="random_id",
            run_key="cipro_positive_random_id",
            target_class="cipro_only",
            workdir=layout.campaign_workdir("cipro_positive_random_id"),
            config_path=layout.campaign_config_path("cipro_positive_random_id"),
            label_input_path=layout.campaign_label_input_path("cipro_positive_random_id"),
            sidecar_path=layout.campaign_sidecar_path("cipro_positive_random_id", "random_id"),
        ),
        RunSpec(
            campaign_key="cipro",
            oracle_id=NULL_ORACLE_ID,
            split_id="random_id",
            run_key="cipro_null_random_id",
            target_class="cipro_only",
            workdir=layout.campaign_workdir("cipro_null_random_id"),
            config_path=layout.campaign_config_path("cipro_null_random_id"),
            label_input_path=layout.campaign_label_input_path("cipro_null_random_id"),
            sidecar_path=layout.campaign_sidecar_path("cipro_null_random_id", "random_id"),
        ),
        RunSpec(
            campaign_key="cipro",
            oracle_id=ORACLE_ID,
            split_id="leave_sigma35_variant",
            run_key="cipro_positive_leave_sigma35_variant",
            target_class="cipro_only",
            workdir=layout.campaign_workdir("cipro_positive_leave_sigma35_variant"),
            config_path=layout.campaign_config_path("cipro_positive_leave_sigma35_variant"),
            label_input_path=layout.campaign_label_input_path("cipro_positive_leave_sigma35_variant"),
            sidecar_path=layout.campaign_sidecar_path("cipro_positive_leave_sigma35_variant", "leave_sigma35_variant"),
        ),
    ]
    plan = ProbePlan(
        run_root=run_root,
        initial_label_count=1,
        selection_k=1,
        seed=7,
        rounds=1,
        gate="all",
        splits=("random_id", "leave_sigma35_variant"),
        apply=True,
        runs=runs,
    )
    labels = pd.DataFrame(
        {
            "oracle_id": [ORACLE_ID] * len(ids),
            "id": ids,
            "sequence": ["AAAA", "CCCC", "GGGG", "TTTT"],
            "axis_class": ["cipro_only"] * len(ids),
            "quality_flag": ["ok"] * len(ids),
            "logic4": [[0, 0, 1, 1]] * len(ids),
            "v00": [0.0] * len(ids),
            "v10": [0.0] * len(ids),
            "v01": [1.0] * len(ids),
            "v11": [1.0] * len(ids),
        }
    )

    import dnadesign.studies.units.stress_ethanol_cipro_growth.opal_densegen_axis_probe.scratch as scratch_mod

    monkeypatch.setattr(scratch_mod, "_write_campaign_config", lambda *args, **kwargs: None)

    materialize_probe_inputs(
        repo_root=tmp_path,
        plan=plan,
        labels=labels,
        null_labels=labels.assign(oracle_id=NULL_ORACLE_ID),
        split_metadata={
            "random_id": {"train_ids": ["random-train"], "eval_ids": ["random-eval"]},
            "leave_sigma35_variant": {"train_ids": ["leave-train"], "eval_ids": ["leave-eval"]},
        },
    )

    assert layout.split_records_path("random_id").is_symlink()
    assert layout.split_records_path("leave_sigma35_variant").is_symlink()
    assert layout.split_records_path("random_id").resolve() == source_records.resolve()
    assert layout.split_records_path("leave_sigma35_variant").resolve() == source_records.resolve()
    random_scope = pd.read_parquet(layout.split_candidate_scope_path("random_id"))
    leave_scope = pd.read_parquet(layout.split_candidate_scope_path("leave_sigma35_variant"))
    assert sorted(random_scope["id"].astype(str).tolist()) == ["random-eval", "random-train"]
    assert sorted(leave_scope["id"].astype(str).tolist()) == ["leave-eval", "leave-train"]
    collection = json.loads(layout.campaign_collection_manifest_path.read_text(encoding="utf-8"))
    assert collection["schema_version"] == "opal.campaign_collection.v1"
    assert collection["dimensions"] == ["target", "label_oracle_kind", "label_family_id", "label_split_id", "seed"]
    assert collection["relationships"] == [
        {
            "kind": "control_pair",
            "left_role": "positive",
            "right_role": "null",
            "match_on": ["target", "label_family_id", "label_split_id", "seed"],
            "replicate_on": ["seed"],
        }
    ]


def test_selected_ids_from_round_rejects_duplicate_selection_ids(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    selection_path = workdir / "outputs" / "rounds" / "round_0" / "selection" / "selection_top_k.csv"
    selection_path.parent.mkdir(parents=True)
    selection_path.write_text("id,score\ncandidate-1,1.0\ncandidate-1,0.9\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="duplicate selected id"):
        selected_ids_from_round("cipro_positive_random_id", workdir, 0)


def test_selected_ids_from_round_rejects_null_selection_ids(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    selection_path = workdir / "outputs" / "rounds" / "round_0" / "selection" / "selection_top_k.csv"
    selection_path.parent.mkdir(parents=True)
    selection_path.write_text("id,score\n,1.0\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="null id"):
        selected_ids_from_round("cipro_positive_random_id", workdir, 0)


def test_selected_ids_from_round_rejects_unexpected_selection_count(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    selection_path = workdir / "outputs" / "rounds" / "round_0" / "selection" / "selection_top_k.csv"
    selection_path.parent.mkdir(parents=True)
    rows = "\n".join(f"candidate-{idx},1.0" for idx in range(7))
    selection_path.write_text(f"id,score\n{rows}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="expected 6 selected"):
        selected_ids_from_round("cipro_positive_random_id", workdir, 0, expected_k=6)


def test_followup_label_input_rejects_duplicate_selected_ids(tmp_path: Path) -> None:
    labels = pd.DataFrame(
        {
            "id": ["candidate-1"],
            "sequence": ["AAAA"],
            "v00": [0.0],
            "v10": [0.0],
            "v01": [1.0],
            "v11": [1.0],
            "y00_star": [0.0],
            "y10_star": [0.0],
            "y01_star": [1.0],
            "y11_star": [1.0],
            "intensity_log2_offset_delta": [0.0],
        }
    )
    run = RunSpec(
        campaign_key="cipro",
        oracle_id=ORACLE_ID,
        split_id="random_id",
        run_key="cipro_positive_random_id",
        target_class="cipro_only",
        workdir=tmp_path / "workdir",
        config_path=tmp_path / "workdir" / "configs" / "campaign.yaml",
        label_input_path=tmp_path / "workdir" / "inputs" / "r0" / "labels-b0.parquet",
        sidecar_path=tmp_path / "workdir" / "observed_labels.parquet",
    )

    with pytest.raises(RuntimeError, match="duplicate ids"):
        write_followup_label_input(
            layout=ProbeArtifactLayout(tmp_path / "probe"),
            run=run,
            labels=labels,
            selected_ids=["candidate-1", "candidate-1"],
            already_labeled=set(),
            round_index=1,
        )


def test_run_opal_rounds_reuses_existing_ingest_and_selection_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "probe"
    workdir = run_root / "scratch_campaigns" / "cipro_positive_random_id"
    run = RunSpec(
        campaign_key="cipro",
        oracle_id=ORACLE_ID,
        split_id="random_id",
        run_key="cipro_positive_random_id",
        target_class="cipro_only",
        workdir=workdir,
        config_path=workdir / "configs" / "campaign.yaml",
        label_input_path=workdir / "inputs" / "r0" / "labels-b0.parquet",
        sidecar_path=workdir / "observed_labels.parquet",
        selection_k=1,
    )
    run.config_path.parent.mkdir(parents=True)
    run.config_path.write_text("campaign_slug: test\n", encoding="utf-8")
    pd.DataFrame(
        {
            "id": ["train-1", "selected-r0"],
            "observed_round": [0, 1],
        }
    ).to_parquet(run.sidecar_path, index=False)
    for round_index, candidate_id in [(0, "selected-r0"), (1, "selected-r1")]:
        selection_path = workdir / "outputs" / "rounds" / f"round_{round_index}" / "selection" / "selection_top_k.csv"
        selection_path.parent.mkdir(parents=True)
        selection_path.write_text(f"id,score\n{candidate_id},1.0\n", encoding="utf-8")
    plan = ProbePlan(
        run_root=run_root,
        initial_label_count=1,
        selection_k=1,
        seed=7,
        rounds=2,
        gate="cipro-random",
        splits=("random_id",),
        apply=True,
        stop_after="status",
        runs=[run],
    )
    commands: list[list[str]] = []

    def fake_run_command(command: list[str], **_: object) -> None:
        commands.append(command)

    import dnadesign.studies.units.stress_ethanol_cipro_growth.opal_densegen_axis_probe.scratch as scratch

    monkeypatch.setattr(scratch, "_run_command", fake_run_command)

    labeled_ids_by_run = run_opal_rounds_for_probe(
        repo_root=tmp_path,
        plan=plan,
        labels_by_oracle={ORACLE_ID: pd.DataFrame(), NULL_ORACLE_ID: pd.DataFrame()},
        split_metadata={"random_id": {"train_ids": ["train-1"]}},
        machine_readable=True,
    )

    opal_subcommands = [command[3] for command in commands if len(command) > 3 and command[:3] == ["uv", "run", "opal"]]
    assert "init" not in opal_subcommands
    assert "ingest-y" not in opal_subcommands
    assert "run" not in opal_subcommands
    assert {"validate", "status"}.issubset(opal_subcommands)
    assert labeled_ids_by_run["cipro_positive_random_id"] == {"train-1", "selected-r0"}


def test_make_training_input_requires_all_train_ids() -> None:
    labels = pd.DataFrame(
        {
            "id": ["id-1"],
            "sequence": ["AAAA"],
            "v00": [0],
            "v10": [0],
            "v01": [0],
            "v11": [0],
            "y00_star": [0.0],
            "y10_star": [0.0],
            "y01_star": [0.0],
            "y11_star": [0.0],
            "intensity_log2_offset_delta": [0.0],
        }
    )

    with pytest.raises(ValueError, match="missing label rows"):
        _make_training_input(labels, ["id-1", "id-missing"])
