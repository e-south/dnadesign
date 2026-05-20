from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dnadesign.opal.tests._cli_helpers import write_campaign_yaml, write_records
from dnadesign.studies.studies.stress_ethanol_cipro_growth.opal_densegen_axis_probe.artifacts import (
    ProbeArtifactLayout,
    RunSpec,
)
from dnadesign.studies.studies.stress_ethanol_cipro_growth.opal_densegen_axis_probe.axis_oracle import (
    build_axis_oracle,
    build_train_ids,
    class_from_logic4,
    derive_axis_label,
    make_permuted_labels,
    parse_sigma35_variant,
)
from dnadesign.studies.studies.stress_ethanol_cipro_growth.opal_densegen_axis_probe.constants import (
    AXIS_CLASS_TO_LOGIC4,
    CANDIDATE_RECORDS,
    NULL_ORACLE_ID,
    ORACLE_ID,
    X_COLUMN,
)
from dnadesign.studies.studies.stress_ethanol_cipro_growth.opal_densegen_axis_probe.decision import (
    _claim_statuses,
    _compact_split_metadata,
    _decision_from_metrics,
    _evaluate_run,
    _persisted_split_metadata,
    _split_metadata_for_all,
)
from dnadesign.studies.studies.stress_ethanol_cipro_growth.opal_densegen_axis_probe.execution import (
    selected_ids_from_round,
    write_followup_label_input,
)
from dnadesign.studies.studies.stress_ethanol_cipro_growth.opal_densegen_axis_probe.paths import (
    validate_run_root_policy,
)
from dnadesign.studies.studies.stress_ethanol_cipro_growth.opal_densegen_axis_probe.plan import (
    build_plan,
    validate_scratch_paths,
)
from dnadesign.studies.studies.stress_ethanol_cipro_growth.opal_densegen_axis_probe.prediction_scoring import (
    predicted_axis_classes,
)
from dnadesign.studies.studies.stress_ethanol_cipro_growth.opal_densegen_axis_probe.progress import (
    summarize_probe_progress,
)
from dnadesign.studies.studies.stress_ethanol_cipro_growth.opal_densegen_axis_probe.review import build_probe_review
from dnadesign.studies.studies.stress_ethanol_cipro_growth.opal_densegen_axis_probe.scratch import (
    _clone_records_file,
    _make_training_input,
    _run_command,
)
from dnadesign.studies.studies.stress_ethanol_cipro_growth.opal_densegen_axis_probe.source_contract import (
    validate_candidate_x_surface,
)
from dnadesign.studies.studies.stress_ethanol_cipro_growth.opal_densegen_axis_probe.status import (
    audit_run_root,
)


def test_probe_package_root_exports_no_flat_api_surface() -> None:
    package = importlib.import_module("dnadesign.studies.studies.stress_ethanol_cipro_growth.opal_densegen_axis_probe")

    assert package.__all__ == []
    assert "build_axis_oracle" not in vars(package)
    assert "main" not in vars(package)


def test_cli_import_keeps_status_path_run_stack_lazy() -> None:
    module = "dnadesign.studies.studies.stress_ethanol_cipro_growth.opal_densegen_axis_probe.cli"
    script = (
        "import sys; "
        f"import {module}; "
        "heavy = sorted(name for name in ('numpy', 'pandas', 'pyarrow', 'yaml') if name in sys.modules); "
        "print(heavy); "
        "raise SystemExit(1 if heavy else 0)"
    )

    result = subprocess.run([sys.executable, "-c", script], check=False, capture_output=True, text=True)

    assert result.returncode == 0, result.stdout + result.stderr


def _detail(*regulators: str) -> list[dict[str, object]]:
    return [
        {"part_kind": "tfbs", "regulator": regulator, "sequence": f"{idx}{regulator}"}
        for idx, regulator in enumerate(regulators)
    ]


def _valid_metrics_payload(runs: list[dict[str, object]] | None = None) -> dict[str, object]:
    return {
        "safety": {
            "path_safety_pass": True,
            "forbidden_input_pass": True,
            "x_surface_pass": True,
            "quality_counts": {"ok": 1},
        },
        "runs": [] if runs is None else runs,
    }


def _write_probe_prediction_campaign(
    workdir: Path,
    predictions: pd.DataFrame,
    *,
    runs: list[tuple[str, int]] | None = None,
) -> Path:
    records_path = workdir / "records.parquet"
    records_path.parent.mkdir(parents=True, exist_ok=True)
    write_records(records_path)
    config_path = workdir / "configs" / "campaign.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    write_campaign_yaml(config_path, workdir=workdir, records_path=records_path)

    run_rows = runs or [("run-0", 0)]
    ledger_dir = workdir / "outputs" / "ledger"
    ledger_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "run_id": [run_id for run_id, _ in run_rows],
            "as_of_round": [round_index for _, round_index in run_rows],
        }
    ).to_parquet(ledger_dir / "runs.parquet", index=False)

    pred = predictions.copy()
    if "run_id" not in pred.columns:
        pred["run_id"] = run_rows[0][0]
    if "as_of_round" not in pred.columns:
        pred["as_of_round"] = run_rows[0][1]
    if "sel__is_selected" not in pred.columns:
        pred["sel__is_selected"] = True
    if "sel__rank_competition" not in pred.columns:
        pred["sel__rank_competition"] = range(1, len(pred) + 1)
    predictions_dir = ledger_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(predictions_dir / "part.parquet", index=False)
    return config_path


@pytest.mark.parametrize(
    ("detail", "expected_class", "expected_vec8"),
    [
        (_detail("background", "background", "background"), "background_only", [0, 0, 0, 0, 0, 0, 0, 0]),
        (_detail("cpxR", "background", "background"), "ethanol_only", [0, 1, 0, 1, 0, 1, 0, 1]),
        (_detail("lexA_CTGTATAWAWWHACA", "background", "background"), "cipro_only", [0, 0, 1, 1, 0, 0, 1, 1]),
        (_detail("baeR", "lexA_CTGTATAWAWWHACA", "background"), "dual_axis_and", [0, 0, 0, 1, 0, 0, 0, 1]),
    ],
)
def test_derive_axis_label_uses_part_detail_not_plan(
    detail: list[dict[str, object]], expected_class: str, expected_vec8: list[int]
) -> None:
    label = derive_axis_label(
        {
            "id": "candidate-1",
            "densegen__used_tfbs_detail": detail,
            "densegen__plan": "background_only__sig35=f",
        }
    )

    assert label.axis_class == expected_class
    assert label.logic4 == AXIS_CLASS_TO_LOGIC4[expected_class]
    assert label.vec8 == expected_vec8


def test_plan_axis_mismatch_is_flagged_without_coercing_label() -> None:
    label = derive_axis_label(
        {
            "id": "candidate-1",
            "densegen__used_tfbs_detail": _detail("lexA_CTGTATAWAWWHACA", "background", "background"),
            "densegen__plan": "ethanol__sig35=f",
        }
    )

    assert label.axis_class == "cipro_only"
    assert label.quality_flag == "plan_axis_mismatch"


def test_missing_part_detail_excludes_row_even_when_plan_is_supported() -> None:
    label = derive_axis_label(
        {
            "id": "candidate-1",
            "densegen__used_tfbs_detail": None,
            "densegen__plan": "ciprofloxacin__sig35=f",
        }
    )

    assert label.axis_class is None
    assert label.quality_flag == "missing_used_tfbs_detail"


def test_malformed_part_detail_excludes_row() -> None:
    label = derive_axis_label(
        {
            "id": "candidate-1",
            "densegen__used_tfbs_detail": [{"regulator": "lexA_CTGTATAWAWWHACA"}],
            "densegen__plan": "ciprofloxacin__sig35=f",
        }
    )

    assert label.axis_class is None
    assert label.quality_flag == "malformed_used_tfbs_detail"


def test_unknown_tfbs_regulator_excludes_row() -> None:
    label = derive_axis_label(
        {
            "id": "candidate-1",
            "densegen__used_tfbs_detail": _detail("surpriseRegulator"),
            "densegen__plan": "background_only__sig35=f",
        }
    )

    assert label.axis_class is None
    assert label.quality_flag == "malformed_used_tfbs_detail"


def test_parse_sigma35_variant_from_densegen_plan_suffix() -> None:
    assert parse_sigma35_variant("ethanol_ciprofloxacin__sig35=d") == "d"
    assert parse_sigma35_variant("ethanol") is None


def test_vectorized_prediction_axis_classes_preserve_vec8_contract() -> None:
    values = [
        [0.0, 0.0, 0.9, 1.0, 0.0, 0.0, 0.9, 1.0],
        [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    ]

    assert predicted_axis_classes(values) == ["cipro_only", "ethanol_only"]

    with pytest.raises(RuntimeError, match="vec8"):
        predicted_axis_classes([[0.0, 1.0, 0.0, 1.0]])


def test_build_axis_oracle_prefers_sidecar_detail_by_id() -> None:
    candidates = pd.DataFrame(
        [
            {"id": "a", "sequence": "AAAA", "densegen__used_tfbs_detail": None, "densegen__plan": "ethanol__sig35=f"},
            {
                "id": "b",
                "sequence": "CCCC",
                "densegen__used_tfbs_detail": _detail("background"),
                "densegen__plan": "background_only__sig35=e",
            },
        ]
    )
    densegen_sidecar = pd.DataFrame(
        [
            {
                "id": "a",
                "densegen__used_tfbs_detail": _detail("cpxR", "background"),
                "densegen__plan": "ethanol__sig35=f",
                "densegen__sampling_library_hash": "hash-a",
            }
        ]
    )

    labels = build_axis_oracle(candidates, densegen_sidecar=densegen_sidecar)

    row_a = labels.set_index("id").loc["a"]
    assert row_a["axis_class"] == "ethanol_only"
    assert row_a["quality_flag"] == "ok"
    assert row_a["sigma35_variant"] == "f"
    assert row_a["densegen__sampling_library_hash"] == "hash-a"


def test_build_axis_oracle_rejects_sidecar_duplicate_ids() -> None:
    candidates = pd.DataFrame(
        [{"id": "a", "sequence": "AAAA", "densegen__used_tfbs_detail": None, "densegen__plan": "ethanol__sig35=f"}]
    )
    densegen_sidecar = pd.DataFrame(
        [
            {"id": "a", "densegen__used_tfbs_detail": _detail("cpxR"), "densegen__plan": "ethanol__sig35=f"},
            {"id": "a", "densegen__used_tfbs_detail": _detail("cpxR"), "densegen__plan": "ethanol__sig35=f"},
        ]
    )

    with pytest.raises(ValueError, match="duplicate id"):
        build_axis_oracle(candidates, densegen_sidecar=densegen_sidecar)


def test_build_axis_oracle_rejects_candidate_sidecar_conflicts() -> None:
    candidates = pd.DataFrame(
        [
            {
                "id": "a",
                "sequence": "AAAA",
                "densegen__used_tfbs_detail": _detail("cpxR"),
                "densegen__plan": "ethanol__sig35=f",
            }
        ]
    )
    densegen_sidecar = pd.DataFrame(
        [
            {
                "id": "a",
                "densegen__used_tfbs_detail": _detail("lexA_CTGTATAWAWWHACA"),
                "densegen__plan": "ethanol__sig35=f",
            }
        ]
    )

    with pytest.raises(ValueError, match="conflict"):
        build_axis_oracle(candidates, densegen_sidecar=densegen_sidecar)


def test_build_train_ids_is_stratified_and_reuses_positive_ids_for_null() -> None:
    rows = []
    for axis_class in AXIS_CLASS_TO_LOGIC4:
        for idx in range(4):
            rows.append(
                {
                    "id": f"{axis_class}-{idx}",
                    "axis_class": axis_class,
                    "quality_flag": "ok",
                    "sigma35_variant": "f" if idx < 2 else "e",
                }
            )
    labels = pd.DataFrame(rows)

    train_ids = build_train_ids(labels, budget=8, seed=7, split_id="random_id")

    selected = labels[labels["id"].isin(train_ids)]
    assert selected.groupby("axis_class").size().to_dict() == {
        "background_only": 2,
        "ethanol_only": 2,
        "cipro_only": 2,
        "dual_axis_and": 2,
    }


def test_build_train_ids_excludes_leave_sigma35_variant_pool() -> None:
    rows = []
    for axis_class in AXIS_CLASS_TO_LOGIC4:
        for variant in ("a", "b", "c"):
            for idx in range(2):
                rows.append(
                    {
                        "id": f"{axis_class}-{variant}-{idx}",
                        "axis_class": axis_class,
                        "quality_flag": "ok",
                        "sigma35_variant": variant,
                    }
                )
    labels = pd.DataFrame(rows)

    train_ids, metadata = build_train_ids(
        labels,
        budget=8,
        seed=7,
        split_id="leave_sigma35_variant",
        return_metadata=True,
    )

    selected = labels[labels["id"].isin(train_ids)]
    assert metadata["heldout_sigma35"] not in set(selected["sigma35_variant"])


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


def test_make_permuted_labels_preserves_distribution_and_changes_alignment() -> None:
    labels = pd.DataFrame(
        {
            "id": [f"id-{idx}" for idx in range(8)],
            "vec8": [[idx % 2] * 8 for idx in range(8)],
            "axis_class": ["background_only", "ethanol_only", "cipro_only", "dual_axis_and"] * 2,
            "quality_flag": ["ok"] * 8,
        }
    )

    permuted = make_permuted_labels(labels, seed=7)

    assert sorted(map(tuple, permuted["vec8"])) == sorted(map(tuple, labels["vec8"]))
    assert not permuted.set_index("id")["vec8"].equals(labels.set_index("id")["vec8"])


def test_make_permuted_labels_keeps_non_ok_rows_unassigned() -> None:
    labels = pd.DataFrame(
        {
            "id": ["ok-a", "ok-b", "bad"],
            "vec8": [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 1, 0, 1],
                None,
            ],
            "axis_class": ["background_only", "ethanol_only", None],
            "quality_flag": ["ok", "ok", "missing_used_tfbs_detail"],
            "v00": [0.0, 0.0, pd.NA],
            "v10": [0.0, 1.0, pd.NA],
            "v01": [0.0, 0.0, pd.NA],
            "v11": [0.0, 1.0, pd.NA],
            "y00_star": [0.0, 0.0, pd.NA],
            "y10_star": [0.0, 1.0, pd.NA],
            "y01_star": [0.0, 0.0, pd.NA],
            "y11_star": [0.0, 1.0, pd.NA],
        }
    )

    permuted = make_permuted_labels(labels, seed=7)

    bad = permuted.set_index("id").loc["bad"]
    assert bad["vec8"] is None
    assert pd.isna(bad["v00"])
    assert pd.isna(bad["axis_class"])


def test_class_from_logic4_uses_nearest_canonical_vector() -> None:
    assert class_from_logic4([0.05, 0.10, 0.85, 0.90]) == "cipro_only"
    assert class_from_logic4([0.10, 0.20, 0.20, 0.75]) == "dual_axis_and"


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
    plan = build_plan(run_root=tmp_path / "probe", budget=96, seed=7, gate="source", splits=("random_id",))

    assert plan.apply is False
    assert plan.runs == []
    assert plan.commands == []


def test_build_plan_stop_after_validate_avoids_scoring_commands(tmp_path: Path) -> None:
    plan = build_plan(
        run_root=tmp_path / "probe",
        budget=96,
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
        budget=96,
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


def test_build_plan_rejects_unknown_stop_stage(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported stop_after"):
        build_plan(
            run_root=tmp_path / "probe",
            budget=96,
            seed=7,
            gate="cipro-random",
            splits=("random_id",),
            stop_after="score-everything-now",
        )


def test_build_plan_rejects_invalid_round_count(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="rounds"):
        build_plan(
            run_root=tmp_path / "probe",
            budget=96,
            seed=7,
            gate="cipro-random",
            splits=("random_id",),
            rounds=0,
        )


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

    with pytest.raises(RuntimeError, match="duplicate ids"):
        write_followup_label_input(
            layout=ProbeArtifactLayout(tmp_path / "probe"),
            run_key="cipro_positive_random_id",
            labels=labels,
            selected_ids=["candidate-1", "candidate-1"],
            already_labeled=set(),
            round_index=1,
        )


def test_audit_run_root_reports_pending_source_gate(tmp_path: Path) -> None:
    run_root = tmp_path / "probe"
    (run_root / "labels").mkdir(parents=True)
    (run_root / "splits").mkdir(parents=True)
    (run_root / "reports").mkdir(parents=True)
    label_frame = pd.DataFrame(
        {
            "oracle_id": [ORACLE_ID],
            "id": ["id-1"],
            "sequence": ["AAAA"],
            "axis_class": ["background_only"],
            "quality_flag": ["ok"],
            "vec8": [[0, 0, 0, 0, 0, 0, 0, 0]],
            "v00": [0.0],
            "v10": [0.0],
            "v01": [0.0],
            "v11": [0.0],
            "y00_star": [0.0],
            "y10_star": [0.0],
            "y01_star": [0.0],
            "y11_star": [0.0],
        }
    )
    label_frame.to_parquet(run_root / "labels" / "densegen_part_axis_vec8.parquet", index=False)
    label_frame.assign(oracle_id=NULL_ORACLE_ID).to_parquet(
        run_root / "labels" / "permuted_densegen_part_axis_vec8.parquet",
        index=False,
    )
    (run_root / "splits" / "split_metadata.json").write_text("{}", encoding="utf-8")
    (run_root / "reports" / "metrics.json").write_text(json.dumps(_valid_metrics_payload()), encoding="utf-8")
    (run_root / "reports" / "decision.md").write_text(
        "# opal_densegen_axis_probe_v0 decision\n\n## Decision\n\nPENDING\n",
        encoding="utf-8",
    )

    audit = audit_run_root(run_root)

    assert audit.status == "ok"
    assert audit.decision == "PENDING"
    assert audit.problems == ()


def test_audit_run_root_rejects_corrupt_label_artifacts(tmp_path: Path) -> None:
    run_root = tmp_path / "probe"
    (run_root / "labels").mkdir(parents=True)
    (run_root / "splits").mkdir(parents=True)
    (run_root / "reports").mkdir(parents=True)
    (run_root / "labels" / "densegen_part_axis_vec8.parquet").write_bytes(b"placeholder")
    (run_root / "labels" / "permuted_densegen_part_axis_vec8.parquet").write_bytes(b"placeholder")
    (run_root / "splits" / "split_metadata.json").write_text("{}", encoding="utf-8")
    (run_root / "reports" / "metrics.json").write_text(json.dumps(_valid_metrics_payload()), encoding="utf-8")
    (run_root / "reports" / "decision.md").write_text(
        "# opal_densegen_axis_probe_v0 decision\n\n## Decision\n\nPENDING\n",
        encoding="utf-8",
    )

    audit = audit_run_root(run_root)

    assert audit.status == "attention"
    assert "densegen_labels_parquet_unreadable" in audit.problems
    assert "null_labels_parquet_unreadable" in audit.problems


def test_audit_run_root_rejects_split_metadata_paths_outside_splits_dir(tmp_path: Path) -> None:
    run_root = tmp_path / "probe"
    (run_root / "splits").mkdir(parents=True)
    (run_root / "splits" / "split_metadata.json").write_text(
        json.dumps(
            {
                "random_id": {
                    "split_id": "random_id",
                    "train_ids_path": "../outside.parquet",
                    "eval_ids_path": "/tmp/outside.parquet",
                }
            }
        ),
        encoding="utf-8",
    )

    audit = audit_run_root(run_root)

    assert audit.status == "attention"
    assert "split_metadata_random_id_train_ids_path_outside_splits_dir" in audit.problems
    assert "split_metadata_random_id_eval_ids_path_outside_splits_dir" in audit.problems


def test_audit_run_root_rejects_malformed_metrics_shape(tmp_path: Path) -> None:
    run_root = tmp_path / "probe"
    (run_root / "reports").mkdir(parents=True)
    (run_root / "reports" / "metrics.json").write_text(
        json.dumps({"safety": {"path_safety_pass": True}, "runs": [42]}),
        encoding="utf-8",
    )

    audit = audit_run_root(run_root)

    assert audit.status == "attention"
    assert "metrics_json_safety_missing_forbidden_input_pass" in audit.problems
    assert "metrics_json_safety_missing_x_surface_pass" in audit.problems
    assert "metrics_json_safety_missing_quality_counts" in audit.problems
    assert "metrics_json_runs_0_not_mapping" in audit.problems


def test_audit_run_root_rejects_invalid_decision_value(tmp_path: Path) -> None:
    run_root = tmp_path / "probe"
    (run_root / "reports").mkdir(parents=True)
    (run_root / "reports" / "metrics.json").write_text(json.dumps(_valid_metrics_payload()), encoding="utf-8")
    (run_root / "reports" / "decision.md").write_text(
        "# opal_densegen_axis_probe_v0 decision\n\n## Decision\n\nBOGUS\n",
        encoding="utf-8",
    )

    audit = audit_run_root(run_root)

    assert audit.status == "attention"
    assert "decision_value_invalid" in audit.problems


def test_audit_run_root_requires_scratch_records_for_planned_campaigns(tmp_path: Path) -> None:
    run_root = tmp_path / "probe"
    (run_root / "labels").mkdir(parents=True)
    (run_root / "splits").mkdir(parents=True)
    (run_root / "reports").mkdir(parents=True)
    (run_root / "scratch_campaigns" / "cipro_positive_random_id").mkdir(parents=True)
    label_frame = pd.DataFrame(
        {
            "oracle_id": [ORACLE_ID],
            "id": ["id-1"],
            "sequence": ["AAAA"],
            "axis_class": ["background_only"],
            "quality_flag": ["ok"],
            "vec8": [[0, 0, 0, 0, 0, 0, 0, 0]],
            "v00": [0.0],
            "v10": [0.0],
            "v01": [0.0],
            "v11": [0.0],
            "y00_star": [0.0],
            "y10_star": [0.0],
            "y01_star": [0.0],
            "y11_star": [0.0],
        }
    )
    label_frame.to_parquet(run_root / "labels" / "densegen_part_axis_vec8.parquet", index=False)
    label_frame.assign(oracle_id=NULL_ORACLE_ID).to_parquet(
        run_root / "labels" / "permuted_densegen_part_axis_vec8.parquet",
        index=False,
    )
    (run_root / "splits" / "split_metadata.json").write_text("{}", encoding="utf-8")
    (run_root / "reports" / "metrics.json").write_text(json.dumps(_valid_metrics_payload()), encoding="utf-8")
    (run_root / "reports" / "decision.md").write_text(
        "# opal_densegen_axis_probe_v0 decision\n\n## Decision\n\nPENDING\n",
        encoding="utf-8",
    )

    audit = audit_run_root(run_root)

    assert audit.status == "attention"
    assert "scratch_records_missing_for_planned_campaigns" in audit.problems


def test_clone_records_file_requires_manifest_for_existing_scratch_file(tmp_path: Path) -> None:
    src = tmp_path / "source.parquet"
    dst = tmp_path / "scratch" / "records.parquet"
    src.write_bytes(b"source-records")
    _clone_records_file(src, dst, copy_mode="full")
    assert (dst.parent / "records_manifest.json").exists()

    _clone_records_file(src, dst, copy_mode="full")
    (dst.parent / "records_manifest.json").unlink()

    with pytest.raises(RuntimeError, match="matching source manifest"):
        _clone_records_file(src, dst, copy_mode="full")


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


def test_validate_candidate_x_surface_checks_schema_and_row_count(tmp_path: Path) -> None:
    records_path = tmp_path / CANDIDATE_RECORDS
    records_path.parent.mkdir(parents=True)
    values = pa.array([0.0] * (2 * 8192), type=pa.float32())
    table = pa.table(
        {
            "id": pa.array(["id-1", "id-2"]),
            X_COLUMN: pa.FixedSizeListArray.from_arrays(values, list_size=8192),
        }
    )
    pq.write_table(table, records_path)

    summary = validate_candidate_x_surface(tmp_path, expected_rows=2)

    assert summary["x_dim"] == 8192
    assert summary["x_value_type"] == "float"
    assert summary["row_count"] == 2
    assert summary["validation_level"] == "parquet_schema_and_row_count"


def test_validate_candidate_x_surface_rejects_wrong_x_dimension(tmp_path: Path) -> None:
    records_path = tmp_path / CANDIDATE_RECORDS
    records_path.parent.mkdir(parents=True)
    values = pa.array([0.0] * 4, type=pa.float32())
    table = pa.table(
        {
            "id": pa.array(["id-1"]),
            X_COLUMN: pa.FixedSizeListArray.from_arrays(values, list_size=4),
        }
    )
    pq.write_table(table, records_path)

    with pytest.raises(ValueError, match="dimension 4"):
        validate_candidate_x_surface(tmp_path, expected_rows=1)


def test_validate_candidate_x_surface_rejects_non_float32_x_values(tmp_path: Path) -> None:
    records_path = tmp_path / CANDIDATE_RECORDS
    records_path.parent.mkdir(parents=True)
    values = pa.array([0] * 8192, type=pa.int32())
    table = pa.table(
        {
            "id": pa.array(["id-1"]),
            X_COLUMN: pa.FixedSizeListArray.from_arrays(values, list_size=8192),
        }
    )
    pq.write_table(table, records_path)

    with pytest.raises(ValueError, match="float32"):
        validate_candidate_x_surface(tmp_path, expected_rows=1)


def test_evaluate_run_rejects_partial_prediction_ledgers(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    config_path = _write_probe_prediction_campaign(
        workdir,
        pd.DataFrame(
            {
                "id": ["eval-1"],
                "pred__y_hat_model": [[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]],
                "pred__score_selected": [1.0],
            }
        ),
    )
    labels = pd.DataFrame(
        {
            "id": ["train-1", "eval-1", "eval-2"],
            "axis_class": ["background_only", "cipro_only", "cipro_only"],
            "quality_flag": ["ok", "ok", "ok"],
        }
    )
    run = RunSpec(
        campaign_key="cipro",
        oracle_id=ORACLE_ID,
        split_id="random_id",
        run_key="cipro_positive_random_id",
        target_class="cipro_only",
        workdir=workdir,
        config_path=config_path,
        label_input_path=workdir / "inputs" / "r0" / "vec8-b0.parquet",
        sidecar_path=workdir / "sidecar.parquet",
    )

    with pytest.raises(RuntimeError, match="missing eval id"):
        _evaluate_run(
            run=run,
            positive_labels=labels,
            run_labels=labels,
            split_metadata={"train_ids": ["train-1"]},
        )


def test_evaluate_run_requires_prediction_schema(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    config_path = _write_probe_prediction_campaign(
        workdir,
        pd.DataFrame({"id": ["eval-1"], "pred__score_selected": [1.0]}),
    )
    labels = pd.DataFrame(
        {
            "id": ["train-1", "eval-1"],
            "axis_class": ["background_only", "cipro_only"],
            "quality_flag": ["ok", "ok"],
        }
    )
    run = RunSpec(
        campaign_key="cipro",
        oracle_id=ORACLE_ID,
        split_id="random_id",
        run_key="cipro_positive_random_id",
        target_class="cipro_only",
        workdir=workdir,
        config_path=config_path,
        label_input_path=workdir / "inputs" / "r0" / "vec8-b0.parquet",
        sidecar_path=workdir / "sidecar.parquet",
    )

    with pytest.raises(RuntimeError, match="missing column"):
        _evaluate_run(
            run=run,
            positive_labels=labels,
            run_labels=labels,
            split_metadata={"train_ids": ["train-1"]},
        )


def test_evaluate_run_rejects_duplicate_prediction_ids(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    config_path = _write_probe_prediction_campaign(
        workdir,
        pd.DataFrame(
            {
                "id": ["eval-1", "eval-1"],
                "pred__y_hat_model": [
                    [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
                    [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                ],
                "pred__score_selected": [1.0, 0.5],
            }
        ),
    )
    labels = pd.DataFrame(
        {
            "id": ["train-1", "eval-1"],
            "axis_class": ["background_only", "cipro_only"],
            "quality_flag": ["ok", "ok"],
        }
    )
    run = RunSpec(
        campaign_key="cipro",
        oracle_id=ORACLE_ID,
        split_id="random_id",
        run_key="cipro_positive_random_id",
        target_class="cipro_only",
        workdir=workdir,
        config_path=config_path,
        label_input_path=workdir / "inputs" / "r0" / "vec8-b0.parquet",
        sidecar_path=workdir / "sidecar.parquet",
    )

    with pytest.raises(RuntimeError, match="duplicate prediction id"):
        _evaluate_run(
            run=run,
            positive_labels=labels,
            run_labels=labels,
            split_metadata={"train_ids": ["train-1"]},
        )


def test_evaluate_run_scores_actual_selected_rows_not_highest_unselected_score(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    config_path = _write_probe_prediction_campaign(
        workdir,
        pd.DataFrame(
            {
                "id": ["eval-high-unselected", "eval-selected"],
                "pred__y_hat_model": [
                    [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
                ],
                "pred__score_selected": [0.99, 0.5],
                "sel__is_selected": [False, True],
                "sel__rank_competition": [2, 1],
            }
        ),
    )
    labels = pd.DataFrame(
        {
            "id": ["train-1", "eval-high-unselected", "eval-selected"],
            "axis_class": ["background_only", "background_only", "cipro_only"],
            "quality_flag": ["ok", "ok", "ok"],
        }
    )
    run = RunSpec(
        campaign_key="cipro",
        oracle_id=ORACLE_ID,
        split_id="random_id",
        run_key="cipro_positive_random_id",
        target_class="cipro_only",
        workdir=workdir,
        config_path=config_path,
        label_input_path=workdir / "inputs" / "r0" / "vec8-b0.parquet",
        sidecar_path=workdir / "sidecar.parquet",
    )

    metrics = _evaluate_run(
        run=run,
        positive_labels=labels,
        run_labels=labels,
        split_metadata={"train_ids": ["train-1"]},
    )

    assert metrics["selected_ids"] == ["eval-selected"]
    assert metrics["selected_target_precision_at_k_true"] == 1.0


def test_evaluate_run_rejects_string_selection_flags(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    config_path = _write_probe_prediction_campaign(
        workdir,
        pd.DataFrame(
            {
                "id": ["eval-1"],
                "pred__y_hat_model": [[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]],
                "pred__score_selected": [0.99],
                "sel__is_selected": ["False"],
                "sel__rank_competition": [1],
            }
        ),
    )
    labels = pd.DataFrame(
        {
            "id": ["train-1", "eval-1"],
            "axis_class": ["background_only", "cipro_only"],
            "quality_flag": ["ok", "ok"],
        }
    )
    run = RunSpec(
        campaign_key="cipro",
        oracle_id=ORACLE_ID,
        split_id="random_id",
        run_key="cipro_positive_random_id",
        target_class="cipro_only",
        workdir=workdir,
        config_path=config_path,
        label_input_path=workdir / "inputs" / "r0" / "vec8-b0.parquet",
        sidecar_path=workdir / "sidecar.parquet",
    )

    with pytest.raises(RuntimeError, match="sel__is_selected must be boolean"):
        _evaluate_run(
            run=run,
            positive_labels=labels,
            run_labels=labels,
            split_metadata={"train_ids": ["train-1"]},
        )


def test_decision_rejects_missing_prediction_metrics() -> None:
    with pytest.raises(ValueError, match="missing_predictions"):
        _decision_from_metrics(
            [
                {
                    "run_key": "cipro_positive_random_id",
                    "campaign": "cipro",
                    "oracle_id": ORACLE_ID,
                    "split_id": "random_id",
                    "status": "missing_predictions",
                }
            ],
            {"path_safety_pass": True, "forbidden_input_pass": True, "x_surface_pass": True},
        )


def test_decision_stops_when_x_surface_contract_fails() -> None:
    decision = _decision_from_metrics(
        [
            {
                "run_key": "cipro_positive_random_id",
                "campaign": "cipro",
                "oracle_id": ORACLE_ID,
                "split_id": "random_id",
                "target_lift_at_k_true": 2.0,
            }
        ],
        {"path_safety_pass": True, "forbidden_input_pass": True, "x_surface_pass": False},
    )

    assert decision == "STOP"


def test_decision_stops_when_null_enriches_true_target_class() -> None:
    decision = _decision_from_metrics(
        [
            {
                "campaign": "cipro",
                "oracle_id": NULL_ORACLE_ID,
                "split_id": "random_id",
                "target_lift_at_k_true": 1.5,
            }
        ],
        {"path_safety_pass": True, "forbidden_input_pass": True, "x_surface_pass": True},
    )

    assert decision == "STOP"


def test_decision_debugs_incomplete_positive_null_pairs() -> None:
    decision = _decision_from_metrics(
        [
            {
                "campaign": "cipro",
                "oracle_id": ORACLE_ID,
                "split_id": "random_id",
                "target_lift_at_k_true": 2.0,
            }
        ],
        {"path_safety_pass": True, "forbidden_input_pass": True, "x_surface_pass": True},
    )

    assert decision == "DEBUG"


def test_decision_pass_is_scoped_to_cipro_random_gate() -> None:
    decision = _decision_from_metrics(
        [
            {
                "campaign": "cipro",
                "oracle_id": ORACLE_ID,
                "split_id": "random_id",
                "target_lift_at_k_true": 3.0,
            },
            {
                "campaign": "cipro",
                "oracle_id": NULL_ORACLE_ID,
                "split_id": "random_id",
                "target_lift_at_k_true": 0.7,
            },
        ],
        {"path_safety_pass": True, "forbidden_input_pass": True, "x_surface_pass": True},
    )

    assert decision == "PASS_CIPRO_RANDOM_GATE"


def test_probe_report_reuses_opal_campaign_review_primitives(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from dnadesign.opal.tests._cli_helpers import write_campaign_yaml, write_ledger, write_records, write_state
    from dnadesign.studies.studies.stress_ethanol_cipro_growth.opal_densegen_axis_probe.cli import main as probe_main

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
    (reports_dir / "metrics.json").write_text(
        json.dumps(
            _valid_metrics_payload(
                [
                    {
                        "run_key": "cipro_positive_random_id",
                        "campaign": "cipro",
                        "oracle_id": ORACLE_ID,
                        "split_id": "random_id",
                        "target_class": "cipro_only",
                        "train_count": 1,
                        "eval_count": 2,
                        "selected_target_precision_at_k_true": 0.5,
                        "target_lift_at_k_true": 2.0,
                        "off_target_class_distribution_true": {
                            "background_only": 0,
                            "ethanol_only": 0,
                            "cipro_only": 1,
                            "dual_axis_and": 0,
                        },
                    }
                ]
            )
        ),
        encoding="utf-8",
    )
    (reports_dir / "decision.md").write_text(
        "# opal_densegen_axis_probe_v0 decision\n\n## Decision\n\nPASS_CIPRO_RANDOM_GATE\n",
        encoding="utf-8",
    )

    payload = build_probe_review(run_root)

    assert Path(payload["review"]).exists()
    assert Path(payload["index"]).exists()
    assert Path(payload["run_manifest"]).exists()
    opal_review = workdir / "outputs" / "review" / "review.md"
    assert opal_review.exists()
    opal_index = workdir / "outputs" / "review" / "index.html"
    assert opal_index.exists()
    review_text = Path(payload["review"]).read_text(encoding="utf-8")
    assert "OPAL campaign run review artifacts remain campaign-scoped" in review_text
    assert "PASS_CIPRO_RANDOM_GATE" in review_text
    index_text = Path(payload["index"]).read_text(encoding="utf-8")
    assert "DenseGen axis probe review" in index_text
    assert "cipro_positive_random_id" in index_text
    assert probe_main(["report", "--run-root", str(run_root), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["decision"] == "DEBUG"


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
    assert payload["campaign_count"] == 1
    campaign = payload["campaigns"][0]
    assert campaign["run_key"] == "cipro_positive_random_id"
    assert campaign["round_index"] == 0
    assert campaign["last_stage"] == "done"
    assert campaign["predict"]["batch"] == 2
    assert probe_main(["progress", "--run-root", str(run_root), "--json"]) == 0
    progress_json = json.loads(capsys.readouterr().out)
    assert progress_json["schema_version"].endswith(".progress.v1")


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
        budget=4,
        seed=7,
        gate="source",
        splits=("random_id", "leave_sigma35_variant"),
    )

    assert _split_metadata_for_all(labels, plan=plan) == {}


def test_claim_statuses_ignore_missing_prediction_rows() -> None:
    statuses = _claim_statuses(
        [
            {
                "campaign": "cipro",
                "oracle_id": ORACLE_ID,
                "split_id": "random_id",
                "status": "missing_predictions",
            }
        ],
        decision="DEBUG",
    )

    assert statuses["H-CIPRO"] == "not evaluated in this run"
