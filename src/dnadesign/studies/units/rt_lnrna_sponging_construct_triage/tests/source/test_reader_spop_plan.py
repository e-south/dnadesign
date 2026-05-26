"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/source/test_reader_spop_plan.py

Reader-derived SPOP planning checks for the RT-lnRNA sponging construct triage
study.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reader_spop_plan import (
    ReaderSpopContractError,
    build_reader_spop_plan,
)


def _write_reader_experiment(
    root: Path,
    *,
    experiment_id: str,
    report_time: float,
    rows: list[dict[str, object]],
    write_manifest: bool = True,
) -> Path:
    experiment = root / "experiments" / "2026" / experiment_id
    artifact = experiment / "outputs" / "artifacts" / "ratio_reporter_normalizer.transform_ratio"
    manifest_dir = experiment / "outputs" / "manifests"
    inputs = experiment / "inputs"
    artifact.mkdir(parents=True)
    manifest_dir.mkdir(parents=True)
    inputs.mkdir(parents=True)
    (experiment / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema": "reader/v7",
                "experiment": {"id": experiment_id},
                "protocol": {
                    "id": "plate_reader/single_reporter_screen",
                    "analysis": {"reporter_channel": "RFP"},
                    "inputs": {
                        "fold_change": {
                            "report_times": [report_time],
                            "time_tolerance": 0.51,
                            "treatment_column": "treatment",
                            "group_by": ["design_id"],
                            "use_global_baseline": True,
                            "global_baseline_value": "0 nm aTc; 0 uM IPTG",
                        }
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    artifact_path = artifact / "df.parquet"
    pq.write_table(pa.Table.from_pylist(rows), artifact_path)
    if write_manifest:
        ratio_record = {
            "schema_version": 3,
            "record_id": "ratio_reporter_normalizer/df",
            "kind": "dataframe_artifact",
            "contract_id": "tidy.v1",
            "content_digest": "sha256:test-ratio",
            "path": "artifacts/ratio_reporter_normalizer.transform_ratio/df.parquet",
            "producer": {
                "id": "ratio_reporter_normalizer",
                "kind": "pipeline",
                "plugin": "transform/ratio",
            },
            "inputs": [{"label": "df", "record": "overflow/df"}],
        }
        (manifest_dir / "records.json").write_text(
            json.dumps(
                {
                    "schema_version": 3,
                    "latest": {"ratio_reporter_normalizer/df": ratio_record},
                    "history": {"ratio_reporter_normalizer/df": [ratio_record]},
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    return experiment


def _ratio_rows(
    *,
    design_id: str,
    time: float,
    z_by_treatment: dict[str, float],
    od_by_treatment: dict[str, float] | None = None,
    replicates: int = 3,
) -> list[dict[str, object]]:
    od_by_treatment = od_by_treatment or {treatment: 1.0 for treatment in z_by_treatment}
    rows: list[dict[str, object]] = []
    for treatment, z_value in z_by_treatment.items():
        for replicate in range(replicates):
            position = f"A{replicate + 1}"
            rows.append(
                {
                    "time": time,
                    "position": position,
                    "channel": "RFP/OD600",
                    "value": z_value,
                    "design_id": design_id,
                    "treatment": treatment,
                    "overflow": False,
                }
            )
            rows.append(
                {
                    "time": time,
                    "position": position,
                    "channel": "OD600",
                    "value": od_by_treatment[treatment],
                    "design_id": design_id,
                    "treatment": treatment,
                    "overflow": False,
                }
            )
    return rows


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


def test_reader_spop_plan_scores_dose_ladder_and_summarizes_controls(tmp_path: Path) -> None:
    reader_root = tmp_path / "reader"
    experiment_id = "20260101_retron_Eco1_26_43_benchmark"
    rows = [
        *_ratio_rows(
            design_id="pES-retron-26; pBbS2c-rfp",
            time=10.0,
            z_by_treatment={
                "0 nm aTc; 0 uM IPTG": 100.0,
                "200 nm aTc; 0 uM IPTG": 500.0,
                "0 nm aTc; 5 uM IPTG": 160.0,
                "0 nm aTc; 50 uM IPTG": 300.0,
                "0 nm aTc; 500 uM IPTG": 460.0,
            },
        ),
        *_ratio_rows(
            design_id="pES-retron-43; pBbS2c-rfp",
            time=10.0,
            z_by_treatment={
                "0 nm aTc; 0 uM IPTG": 100.0,
                "200 nm aTc; 0 uM IPTG": 500.0,
                "0 nm aTc; 5 uM IPTG": 90.0,
                "0 nm aTc; 50 uM IPTG": 120.0,
                "0 nm aTc; 500 uM IPTG": 115.0,
            },
        ),
    ]
    _write_reader_experiment(reader_root, experiment_id=experiment_id, report_time=10.0, rows=rows)

    plan = build_reader_spop_plan(reader_root=reader_root, experiment_ids=(experiment_id,))

    assert plan.ok, plan.issues
    assert len(plan.observations) == 2
    by_key = {row.candidate_key: row for row in plan.observations}
    assert by_key["retron26"].assay_subject_key == "retron26"
    assert by_key["retron26"].construct_subject_id == "rt_lnrna_pair__eco1_wt_rt__retron26_lnrna__tetO"
    assert by_key["retron26"].proposed_construct_subject_id == by_key["retron26"].construct_subject_id
    assert by_key["retron26"].construct_subject_bridge_status == "resolved_construct_sequence_authority"
    assert by_key["retron26"].reader_artifact_record_id == "ratio_reporter_normalizer/df"
    assert by_key["retron26"].reader_artifact_content_digest == "sha256:test-ratio"
    assert by_key["retron26"].metric_id == "reader_spop_endpoint_auc_v1"
    assert by_key["retron26"].reporter_plasmid_id == "pBbS2c-RFP"
    assert by_key["retron26"].payload_program_id == "rt_lnrna_sponging"
    assert by_key["retron26"].batch_id == experiment_id
    assert by_key["retron26"].replicate_count == 3
    assert by_key["retron26"].uncertainty is None
    assert by_key["retron26"].assay_metadata["lambda_viability"] == 0.5
    assert by_key["retron26"].spop_score_raw == by_key["retron26"].spop_score
    assert by_key["retron26"].spop_score_calibrated is None
    assert {
        "assay_subject_key",
        "proposed_construct_subject_id",
        "construct_subject_id",
        "construct_subject_bridge_status",
        "reader_artifact_ref",
        "payload_program_id",
        "batch_id",
        "replicate_count",
        "uncertainty",
        "assay_metadata",
    } <= set(by_key["retron26"].to_dict())
    assert by_key["retron26"].construct_promotable is True
    assert by_key["retron43"].construct_promotable is True
    assert by_key["retron26"].iptg_doses_uM == (5.0, 50.0, 500.0)
    assert by_key["retron26"].y_derepression_by_dose == pytest.approx((0.15, 0.5, 0.9))
    assert by_key["retron26"].spop_score == pytest.approx((0.15 + 0.5 + 0.9) / 3.0)
    assert by_key["retron26"].spop_score > by_key["retron43"].spop_score
    assert by_key["retron43"].raw_value < by_key["retron43"].normalized_value
    assert "derepression_below_zero_inducer" in by_key["retron43"].qc_flags

    summaries = {row.candidate_key: row for row in plan.candidate_summaries}
    assert summaries["retron26"].observation_count == 1
    assert summaries["retron26"].spop_score_median == pytest.approx(by_key["retron26"].spop_score)
    assert summaries["retron43"].spop_score_median < 0.1


def test_reader_spop_plan_marks_unresolved_variant_sequence_authority(tmp_path: Path) -> None:
    reader_root = tmp_path / "reader"
    experiment_id = "20260102_retron_Eco1_176_benchmark"
    rows = _ratio_rows(
        design_id="pES-retron-176; pBbS2c-rfp",
        time=10.0,
        z_by_treatment={
            "0 nm aTc; 0 uM IPTG": 100.0,
            "200 nm aTc; 0 uM IPTG": 500.0,
            "0 nm aTc; 500 uM IPTG": 520.0,
        },
    )
    _write_reader_experiment(reader_root, experiment_id=experiment_id, report_time=10.0, rows=rows)

    plan = build_reader_spop_plan(reader_root=reader_root, experiment_ids=(experiment_id,))

    assert plan.ok, plan.issues
    observation = plan.observations[0]
    assert observation.candidate_key == "retron176"
    assert observation.construct_promotable is True
    assert observation.construct_subject_bridge_status == "resolved_construct_sequence_authority"
    assert observation.construct_subject_id == "rt_lnrna_pair__eco1_wt_rt__retron176_lnrna__tetO"
    assert observation.proposed_construct_subject_id == observation.construct_subject_id
    assert "construct_sequence_authority_missing" not in observation.qc_flags


def test_reader_spop_plan_rejects_malformed_treatment_for_retron_rows(tmp_path: Path) -> None:
    reader_root = tmp_path / "reader"
    experiment_id = "20260103_retron_Eco1_26_bad_treatment"
    rows = _ratio_rows(
        design_id="pES-retron-26; pBbS2c-rfp",
        time=10.0,
        z_by_treatment={
            "0 nm aTc; 0 uM IPTG": 100.0,
            "200 nm aTc; 0 uM IPTG": 500.0,
            "IPTG high": 450.0,
        },
    )
    _write_reader_experiment(reader_root, experiment_id=experiment_id, report_time=10.0, rows=rows)

    with pytest.raises(ReaderSpopContractError, match="malformed treatment"):
        build_reader_spop_plan(reader_root=reader_root, experiment_ids=(experiment_id,))


def test_reader_spop_plan_reports_endpoint_drift_without_fabricating_label(tmp_path: Path) -> None:
    reader_root = tmp_path / "reader"
    experiment_id = "20260104_retron_Eco1_26_endpoint_drift"
    rows = _ratio_rows(
        design_id="pES-retron-26; pBbS2c-rfp",
        time=0.0,
        z_by_treatment={
            "0 nm aTc; 0 uM IPTG": 100.0,
            "200 nm aTc; 0 uM IPTG": 500.0,
            "0 nm aTc; 500 uM IPTG": 450.0,
        },
    )
    _write_reader_experiment(reader_root, experiment_id=experiment_id, report_time=10.0, rows=rows)

    plan = build_reader_spop_plan(reader_root=reader_root, experiment_ids=(experiment_id,), strict=False)

    assert not plan.ok
    assert plan.observations == ()
    assert plan.issues[0].code == "configured_endpoint_time_absent"
    assert experiment_id in plan.issues[0].message

    with pytest.raises(ReaderSpopContractError, match="configured_endpoint_time_absent"):
        build_reader_spop_plan(reader_root=reader_root, experiment_ids=(experiment_id,), strict=True)


def test_reader_spop_plan_accepts_known_single_point_rt_variant_artifact(tmp_path: Path) -> None:
    reader_root = tmp_path / "reader"
    experiment_id = "20251105_retron_Eco1_RT_variants"
    rows = _ratio_rows(
        design_id="pES-retron-49; pBbS2c-rfp",
        time=0.0,
        z_by_treatment={
            "0 nm aTc; 0 uM IPTG": 100.0,
            "200 nm aTc; 0 uM IPTG": 500.0,
            "0 nm aTc; 500 uM IPTG": 300.0,
        },
    )
    _write_reader_experiment(reader_root, experiment_id=experiment_id, report_time=12.0, rows=rows)

    plan = build_reader_spop_plan(reader_root=reader_root, experiment_ids=(experiment_id,), strict=True)

    assert plan.ok
    assert plan.issues == ()
    observation = plan.observations[0]
    assert observation.candidate_key == "retron49"
    assert observation.endpoint_time_h == 10.0
    assert observation.assay_metadata["reader_artifact_time_h"] == 0.0
    assert observation.assay_metadata["configured_report_time_h"] == 12.0
    assert observation.assay_metadata["endpoint_time_basis"] == "single_point_mid_log_elapsed_time_override"
    assert "single_point_endpoint_time_override" in observation.qc_flags


def test_reader_spop_plan_omits_known_no_strain_retron176_wells(tmp_path: Path) -> None:
    reader_root = tmp_path / "reader"
    experiment_id = "20260507_retron_Eco1_26_43_172_173_174_175_176_benchmark"
    rows = [
        *_ratio_rows(
            design_id="pES-retron-26; pBbS2c-rfp",
            time=10.0,
            z_by_treatment={
                "0 nm aTc; 0 uM IPTG": 100.0,
                "200 nm aTc; 0 uM IPTG": 500.0,
                "0 nm aTc; 500 uM IPTG": 460.0,
            },
        ),
        *_ratio_rows(
            design_id="pES-retron-176; pBbS2c-rfp",
            time=10.0,
            z_by_treatment={
                "0 nm aTc; 0 uM IPTG": 500.0,
                "200 nm aTc; 0 uM IPTG": 450.0,
                "0 nm aTc; 500 uM IPTG": 700.0,
            },
        ),
    ]
    _write_reader_experiment(reader_root, experiment_id=experiment_id, report_time=10.0, rows=rows)

    plan = build_reader_spop_plan(reader_root=reader_root, experiment_ids=(experiment_id,), strict=True)

    assert plan.ok
    assert [row.candidate_key for row in plan.observations] == ["retron26"]
    assert [row.candidate_key for row in plan.candidate_summaries] == ["retron26"]
    assert len(plan.issues) == 1
    assert plan.issues[0].severity == "warning"
    assert plan.issues[0].code == "assay_subject_excluded_no_strain"
    assert "retron176" in plan.issues[0].message


def test_reader_spop_plan_requires_reader_records_manifest(tmp_path: Path) -> None:
    reader_root = tmp_path / "reader"
    experiment_id = "20260105_retron_Eco1_26_missing_manifest"
    rows = _ratio_rows(
        design_id="pES-retron-26; pBbS2c-rfp",
        time=10.0,
        z_by_treatment={
            "0 nm aTc; 0 uM IPTG": 100.0,
            "200 nm aTc; 0 uM IPTG": 500.0,
            "0 nm aTc; 500 uM IPTG": 450.0,
        },
    )
    _write_reader_experiment(
        reader_root,
        experiment_id=experiment_id,
        report_time=10.0,
        rows=rows,
        write_manifest=False,
    )

    with pytest.raises(ReaderSpopContractError, match="records.json"):
        build_reader_spop_plan(reader_root=reader_root, experiment_ids=(experiment_id,))


def test_reader_spop_plan_rejects_positive_control_below_baseline(tmp_path: Path) -> None:
    reader_root = tmp_path / "reader"
    experiment_id = "20260106_retron_Eco1_26_bad_positive_control"
    rows = _ratio_rows(
        design_id="pES-retron-26; pBbS2c-rfp",
        time=10.0,
        z_by_treatment={
            "0 nm aTc; 0 uM IPTG": 500.0,
            "200 nm aTc; 0 uM IPTG": 450.0,
            "0 nm aTc; 500 uM IPTG": 700.0,
        },
    )
    _write_reader_experiment(reader_root, experiment_id=experiment_id, report_time=10.0, rows=rows)

    plan = build_reader_spop_plan(reader_root=reader_root, experiment_ids=(experiment_id,))

    assert not plan.ok
    assert plan.observations == ()
    assert plan.issues[0].code == "positive_control_not_above_baseline"


def test_reader_spop_contract_docs_route_label_materialization_without_opal_objective_drift() -> None:
    repo_root = _repo_root()
    schema_path = (
        repo_root / "docs/studies/rt_lnrna_sponging_construct_triage/operations/contract/schemas/"
        "sponging-assay-observation.schema.yaml"
    )
    schema = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    assert schema["label_contract"]["metric_id"] == "reader_spop_endpoint_auc_v1"
    assert schema["label_contract"]["score_direction"] == "maximize"
    assert schema["label_contract"]["y_expected_length"] == 1
    assert schema["label_contract"]["opal_objective"] == "scalar_identity_v1/scalar"
    assert "spop_score_raw" in schema["derived_fields"]
    assert "spop_score_calibrated" in schema["derived_fields"]
    assert "reader_design_id" in schema["required_fields"]
    assert "construct_subject_bridge_status" in schema["required_fields"]

    contract_doc = repo_root / "docs/studies/rt_lnrna_sponging_construct_triage/contexts/reader-spop-label-contract.md"
    assert contract_doc.exists()
    contract_text = contract_doc.read_text(encoding="utf-8")
    assert "pBbS2c-RFP" in contract_text
    assert "scalar_identity_v1/scalar" in contract_text
    assert "must not run OPAL `spop_v1`" in contract_text

    ops = yaml.safe_load(
        (repo_root / "docs/studies/rt_lnrna_sponging_construct_triage/operations/ops.study.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert "contract/readiness/checks/reader_spop_label_materialization.yaml" in ops["parts"]["preflight"]

    pipeline = yaml.safe_load(
        (
            repo_root
            / "docs/studies/rt_lnrna_sponging_construct_triage/operations/runtime/command-groups/pipeline.yaml"
        ).read_text(encoding="utf-8")
    )
    assert any(group["id"] == "reader_spop_label_materialization" for group in pipeline["command_groups"])
