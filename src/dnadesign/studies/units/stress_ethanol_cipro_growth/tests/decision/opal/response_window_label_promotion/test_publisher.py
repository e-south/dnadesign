"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/response_window_label_promotion/test_publisher.py

End-to-end contract tests for study-owned OPAL label promotion.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from inspect import signature
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from dnadesign.opal import (
    CampaignAnalysis,
    ObservedLabelPromotionBinding,
    load_plot_artifact_manifest,
    verify_observed_label_snapshot,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_window_label_promotion import (
    DEFAULT_OUTPUT_DIRECTORY,
    ResponseWindowLabelPromotionError,
    publish_response_window_labels,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_window_label_promotion import (
    contracts as promotion_contracts,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_window_label_promotion import (
    cumulative as cumulative_promotion,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_window_label_promotion import (
    lineage as promotion_lineage,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_window_label_promotion import (
    materialization as promotion_materialization,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_window_label_promotion import (
    publication as promotion_publication,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_window_label_promotion import (
    publisher as publisher_module,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations import (
    materialize_response_window_observations,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.tests.response_window_observations.test_artifact import (
    _evidence,
)


def test_verified_observations_publish_exact_one_dimensional_opal_labels(tmp_path: Path) -> None:
    observation_bundle = _observation_bundle(tmp_path)
    dataset = _dataset(tmp_path)

    result = publish_response_window_labels(
        observation_bundle_dir=observation_bundle,
        dataset_root=dataset,
    )

    labels = pd.read_parquet(result.label_path)
    assert labels.columns.tolist() == [
        "id",
        "display_label",
        "observed_round",
        "batch_id",
        "y_space",
        "y_obs",
    ]
    assert labels["id"].tolist() == ["candidate-a"]
    assert labels["display_label"].tolist() == ["Candidate A"]
    assert labels["y_obs"].map(lambda value: getattr(value, "ndim", 1)).tolist() == [1]
    assert labels["y_obs"].map(len).tolist() == [8]
    assert result.output_directory.relative_to(dataset).as_posix() == DEFAULT_OUTPUT_DIRECTORY

    binding = ObservedLabelPromotionBinding(
        dataset_root=dataset,
        manifest_path=f"{DEFAULT_OUTPUT_DIRECTORY}/promotion.manifest.json",
        label_path=f"{DEFAULT_OUTPUT_DIRECTORY}/observed_labels.parquet",
        campaign_slug="secg_msrb_greedy",
        study_id="stress_ethanol_cipro_growth",
        y_space="reader_response_window_vector_v1",
    )
    verified = verify_observed_label_snapshot(binding, expected_y_width=8)
    assert verified.promotion.row_count == 1
    assert verified.promotion.candidate_path == (dataset / "records.parquet").resolve()
    provenance = json.loads(result.study_provenance_path.read_text(encoding="utf-8"))
    assert provenance["observation_bundle"]["schema_id"].endswith("response_window_observations.v2")
    assert provenance["schema_id"].endswith("response_window_label_promotion.v5")
    assert provenance["prior_promotion"] is None


def test_cumulative_publication_carries_prior_rows_and_appends_one_later_batch(tmp_path: Path) -> None:
    first_bundle = _observation_bundle(
        tmp_path / "first",
        observed_round=0,
        batch_id="batch_0",
    )
    second_bundle = _observation_bundle(
        tmp_path / "second",
        candidate_id="candidate-b",
        display_label="Candidate B",
        sequence="CCCC",
        observed_round=1,
        batch_id="batch_1",
    )
    dataset = _dataset(tmp_path)
    first = publish_response_window_labels(
        observation_bundle_dir=first_bundle,
        dataset_root=dataset,
        output_relative_directory="_opal/response_window_labels_batch0_v3",
        prior_promotion_manifest_path=None,
    )
    prior_labels = pd.read_parquet(first.label_path)

    second = publish_response_window_labels(
        observation_bundle_dir=second_bundle,
        dataset_root=dataset,
        output_relative_directory="_opal/response_window_labels_batch1_v3",
        prior_promotion_manifest_path=first.promotion_manifest_path,
    )

    cumulative = pd.read_parquet(second.label_path)
    pd.testing.assert_frame_equal(cumulative.iloc[: len(prior_labels)].reset_index(drop=True), prior_labels)
    assert cumulative[["id", "observed_round", "batch_id"]].to_dict(orient="records") == [
        {"id": "candidate-a", "observed_round": 0, "batch_id": "batch_0"},
        {"id": "candidate-b", "observed_round": 1, "batch_id": "batch_1"},
    ]
    provenance = json.loads(second.study_provenance_path.read_text(encoding="utf-8"))
    assert provenance["prior_promotion"] == {
        "label_path": first.label_path.relative_to(dataset).as_posix(),
        "label_sha256": hashlib.sha256(first.label_path.read_bytes()).hexdigest(),
        "manifest_path": first.promotion_manifest_path.relative_to(dataset).as_posix(),
        "manifest_sha256": hashlib.sha256(first.promotion_manifest_path.read_bytes()).hexdigest(),
        "label_event_count": 1,
        "unique_candidate_count": 1,
        "max_observed_round": 0,
    }
    assert provenance["label_contract"]["batch_ids"] == ["batch_0", "batch_1"]
    assert provenance["label_contract"]["observed_rounds"] == [0, 1]


def test_two_batch_publication_survives_opal_run_and_batch_toggle_contract(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    first = publish_response_window_labels(
        observation_bundle_dir=_observation_bundle(tmp_path / "first", observed_round=0, batch_id="batch_0"),
        dataset_root=dataset,
        output_relative_directory="_opal/response_window_labels_batch0_v3",
    )
    cumulative_directory = "_opal/response_window_labels_batch1_v3"
    campaign = _runnable_campaign_config(
        tmp_path / "campaign-config",
        dataset=dataset,
        label_directory=cumulative_directory,
    )
    publish_response_window_labels(
        observation_bundle_dir=_observation_bundle(
            tmp_path / "second",
            observed_round=1,
            batch_id="batch_1",
        ),
        dataset_root=dataset,
        output_relative_directory=cumulative_directory,
        prior_promotion_manifest_path=first.promotion_manifest_path,
    )

    validate = _run_opal("--no-color", "validate", "-c", str(campaign), "--json")
    assert validate.returncode == 0, validate.stdout + validate.stderr
    assert json.loads(validate.stdout)["ok"] is True
    initialize = _run_opal("--no-color", "init", "-c", str(campaign))
    assert initialize.returncode == 0, initialize.stdout + initialize.stderr
    run = _run_opal("--no-color", "run", "-c", str(campaign), "--round", "1", "--json")
    assert run.returncode == 0, run.stdout + run.stderr
    run_payload = json.loads(run.stdout)
    assert run_payload["trained_on"] == 1

    workdir = Path(yaml.safe_load(campaign.read_text(encoding="utf-8"))["campaign"]["workdir"])
    observed = (
        CampaignAnalysis.from_config_path(campaign)
        .read_run_observed_events(
            round_selector=1,
            run_id=run_payload["run_id"],
        )
        .to_pandas()
    )
    assert observed[["id", "observed_round", "batch_id"]].to_dict(orient="records") == [
        {"id": "candidate-a", "observed_round": 0, "batch_id": "batch_0"},
        {"id": "candidate-a", "observed_round": 1, "batch_id": "batch_1"},
    ]

    plot = _run_opal(
        "--no-color",
        "plot",
        "-c",
        str(campaign),
        "--round",
        "1",
        "--run-id",
        run_payload["run_id"],
        "--view",
        "ethanol",
        "--name",
        "msrb_family_frontier",
    )
    assert plot.returncode == 0, plot.stdout + plot.stderr
    manifest_path = workdir / "outputs/plots/msrb_family_frontier_r1.manifest.json"
    manifest = load_plot_artifact_manifest(manifest_path)
    view = manifest["metadata"]["notebook_view"]
    assert view["adapter"] == "layered_scatter_v1"
    assert view["batch_column"] == "batch_key"
    tidy = pd.read_csv(manifest["tidy_csv"])
    observed_rows = tidy.loc[tidy[view["record_kind_column"]].astype(str).eq(view["observed_value"])]
    assert sorted(observed_rows[view["batch_column"]].astype(str).unique()) == ["batch_0", "batch_1"]
    for batch_id in ("batch_0", "batch_1"):
        visible = observed_rows.loc[observed_rows[view["batch_column"]].astype(str).eq(batch_id)]
        assert visible[view["batch_column"]].astype(str).tolist() == [batch_id]


@pytest.mark.parametrize("y_obs", [[1.0] * 7, [1.0] * 7 + [float("inf")]])
def test_cumulative_label_contract_requires_finite_width_eight_vectors(y_obs: list[float]) -> None:
    frame = pd.DataFrame(
        {
            "id": ["candidate-a"],
            "display_label": ["Candidate A"],
            "observed_round": [0],
            "batch_id": ["batch_0"],
            "y_space": ["reader_response_window_vector_v1"],
            "y_obs": [y_obs],
        }
    )

    with pytest.raises(ResponseWindowLabelPromotionError, match="finite one-dimensional.*width 8"):
        cumulative_promotion.extend_label_frame(None, frame)


@pytest.mark.parametrize(
    ("candidate_id", "display_label", "observed_round", "batch_id", "message"),
    [
        ("candidate-a", "Candidate A", 0, "batch_0_repeat", "duplicate candidate/round"),
        ("candidate-b", "Candidate B", 0, "batch_1", "strictly later"),
        ("candidate-b", "Candidate B", 1, "batch_0", "reuses an existing"),
    ],
)
def test_cumulative_publication_rejects_nonappend_batch_semantics(
    tmp_path: Path,
    candidate_id: str,
    display_label: str,
    observed_round: int,
    batch_id: str,
    message: str,
) -> None:
    dataset = _dataset(tmp_path)
    first = publish_response_window_labels(
        observation_bundle_dir=_observation_bundle(tmp_path / "first", observed_round=0, batch_id="batch_0"),
        dataset_root=dataset,
        output_relative_directory="_opal/response_window_labels_batch0_v3",
    )
    sequence = "ACGT" if candidate_id == "candidate-a" else "CCCC"
    second_bundle = _observation_bundle(
        tmp_path / "second",
        candidate_id=candidate_id,
        display_label=display_label,
        sequence=sequence,
        observed_round=observed_round,
        batch_id=batch_id,
    )

    with pytest.raises(ResponseWindowLabelPromotionError, match=message):
        publish_response_window_labels(
            observation_bundle_dir=second_bundle,
            dataset_root=dataset,
            output_relative_directory="_opal/response_window_labels_batch1_v3",
            prior_promotion_manifest_path=first.promotion_manifest_path,
        )

    assert not (dataset / "_opal/response_window_labels_batch1_v3").exists()


def test_cumulative_publication_rejects_prior_label_artifact_drift(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    first = publish_response_window_labels(
        observation_bundle_dir=_observation_bundle(tmp_path / "first", observed_round=0, batch_id="batch_0"),
        dataset_root=dataset,
        output_relative_directory="_opal/response_window_labels_batch0_v3",
    )
    prior_labels = pd.read_parquet(first.label_path)
    prior_labels.loc[0, "display_label"] = "Tampered A"
    prior_labels.to_parquet(first.label_path, index=False)

    with pytest.raises(ResponseWindowLabelPromotionError, match="label artifact SHA-256"):
        publish_response_window_labels(
            observation_bundle_dir=_observation_bundle(
                tmp_path / "second",
                candidate_id="candidate-b",
                display_label="Candidate B",
                sequence="CCCC",
                observed_round=1,
                batch_id="batch_1",
            ),
            dataset_root=dataset,
            output_relative_directory="_opal/response_window_labels_batch1_v3",
            prior_promotion_manifest_path=first.promotion_manifest_path,
        )

    assert not (dataset / "_opal/response_window_labels_batch1_v3").exists()


def test_cumulative_publication_rejects_prior_candidate_sequence_drift(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    first = publish_response_window_labels(
        observation_bundle_dir=_observation_bundle(tmp_path / "first", observed_round=0, batch_id="batch_0"),
        dataset_root=dataset,
        output_relative_directory="_opal/response_window_labels_batch0_v3",
    )
    records_path = dataset / "records.parquet"
    records = pd.read_parquet(records_path)
    records.loc[records["id"].eq("candidate-a"), "sequence"] = "TGCA"
    records.to_parquet(records_path, index=False)

    with pytest.raises(ResponseWindowLabelPromotionError, match="candidate artifact SHA-256"):
        publish_response_window_labels(
            observation_bundle_dir=_observation_bundle(
                tmp_path / "second",
                candidate_id="candidate-b",
                display_label="Candidate B",
                sequence="CCCC",
                observed_round=1,
                batch_id="batch_1",
            ),
            dataset_root=dataset,
            output_relative_directory="_opal/response_window_labels_batch1_v3",
            prior_promotion_manifest_path=first.promotion_manifest_path,
        )

    assert not (dataset / "_opal/response_window_labels_batch1_v3").exists()


def test_cumulative_publication_preserves_prior_exclusions_and_rejects_reason_drift(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    exclusion = [{"candidate_id": "candidate-b", "reason": "nonexact_primary_component"}]
    first_directory = "_opal/response_window_labels_batch0_v3"
    first = publish_response_window_labels(
        observation_bundle_dir=_observation_bundle(tmp_path / "first", with_excluded_candidate=True),
        dataset_root=dataset,
        output_relative_directory=first_directory,
    )
    drift_bundle = _observation_bundle(
        tmp_path / "drift",
        with_excluded_candidate=True,
        candidate_id="candidate-unmeasured",
        display_label="Candidate Unmeasured",
        sequence="AAAA",
        observed_round=1,
        batch_id="batch_1_drift",
        excluded_reason="repeat_source_disagreement",
    )
    with pytest.raises(ResponseWindowLabelPromotionError, match="exclusion reason drift"):
        publish_response_window_labels(
            observation_bundle_dir=drift_bundle,
            dataset_root=dataset,
            output_relative_directory="_opal/response_window_labels_batch1_drift_v3",
            prior_promotion_manifest_path=first.promotion_manifest_path,
        )

    conflict_bundle = _observation_bundle(
        tmp_path / "conflict",
        with_excluded_candidate=True,
        candidate_id="candidate-unmeasured",
        display_label="Candidate Unmeasured",
        sequence="AAAA",
        observed_round=1,
        batch_id="batch_1_conflict",
        excluded_candidate_id="candidate-a",
    )
    with pytest.raises(ResponseWindowLabelPromotionError, match="conflicts with a promoted label"):
        publish_response_window_labels(
            observation_bundle_dir=conflict_bundle,
            dataset_root=dataset,
            output_relative_directory="_opal/response_window_labels_batch1_conflict_v3",
            prior_promotion_manifest_path=first.promotion_manifest_path,
        )

    second_directory = "_opal/response_window_labels_batch1_v3"
    second = publish_response_window_labels(
        observation_bundle_dir=_observation_bundle(
            tmp_path / "second",
            candidate_id="candidate-unmeasured",
            display_label="Candidate Unmeasured",
            sequence="AAAA",
            observed_round=1,
            batch_id="batch_1",
        ),
        dataset_root=dataset,
        output_relative_directory=second_directory,
        prior_promotion_manifest_path=first.promotion_manifest_path,
    )
    provenance = json.loads(second.study_provenance_path.read_text(encoding="utf-8"))
    assert provenance["candidate_selection_exclusions"]["entries"] == exclusion

    third_directory = "_opal/response_window_labels_batch2_v3"
    third = publish_response_window_labels(
        observation_bundle_dir=_observation_bundle(
            tmp_path / "third",
            candidate_id="candidate-b",
            display_label="Candidate B",
            sequence="CCCC",
            observed_round=2,
            batch_id="batch_2",
        ),
        dataset_root=dataset,
        output_relative_directory=third_directory,
        prior_promotion_manifest_path=second.promotion_manifest_path,
    )
    third_provenance = json.loads(third.study_provenance_path.read_text(encoding="utf-8"))
    assert third_provenance["candidate_selection_exclusions"]["entries"] == []


def test_publisher_reads_only_candidate_identity_columns(monkeypatch, tmp_path: Path) -> None:
    observation_bundle = _observation_bundle(tmp_path)
    dataset = _dataset(tmp_path)
    records_path = (dataset / "records.parquet").resolve()
    original_read_parquet = publisher_module.pd.read_parquet
    candidate_column_reads: list[list[str] | None] = []

    def tracked_read_parquet(path, *args, **kwargs):
        if Path(path).resolve() == records_path:
            columns = kwargs.get("columns")
            candidate_column_reads.append(None if columns is None else list(columns))
        return original_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(publisher_module.pd, "read_parquet", tracked_read_parquet)

    publish_response_window_labels(
        observation_bundle_dir=observation_bundle,
        dataset_root=dataset,
    )

    assert candidate_column_reads == [["id", "sequence"]]


def test_promotion_binds_observation_exclusions_and_requires_exact_campaign_parity(tmp_path: Path) -> None:
    observation_bundle = _observation_bundle(tmp_path, with_excluded_candidate=True)
    dataset = _dataset(tmp_path)
    campaign = _campaign_config(
        tmp_path,
        dataset=dataset,
        entries=[{"candidate_id": "candidate-b", "reason": "nonexact_primary_component"}],
    )

    result = publish_response_window_labels(
        observation_bundle_dir=observation_bundle,
        dataset_root=dataset,
    )

    provenance = json.loads(result.study_provenance_path.read_text(encoding="utf-8"))
    assert provenance["candidate_selection_exclusions"] == {
        "authority": "study_observation_bundle",
        "derivation": "contribution_candidates_absent_from_observations",
        "entry_count": 1,
        "entries": [{"candidate_id": "candidate-b", "reason": "nonexact_primary_component"}],
        "exclusion_set_id": "stress_response_window_observation_dispositions_v1",
        "source_record": "contributions",
    }
    promotion_publication.verify_campaign_binding(
        dataset,
        relative_dir=promotion_contracts.confined_relative_directory(DEFAULT_OUTPUT_DIRECTORY),
        expected_width=8,
        campaign_config_path=campaign,
    )


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        ([], "missing candidate_id_exclusion"),
        (
            [
                {"candidate_id": "candidate-b", "reason": "nonexact_primary_component"},
                {"candidate_id": "stale-candidate", "reason": "repeat_excluded_noncomparable"},
            ],
            "extra or stale",
        ),
        (
            [{"candidate_id": "candidate-b", "reason": "repeat_excluded_noncomparable"}],
            "reason mismatch",
        ),
    ],
)
def test_explicit_campaign_binding_rejects_campaign_exclusion_drift(
    tmp_path: Path,
    entries: list[dict[str, str]],
    message: str,
) -> None:
    observation_bundle = _observation_bundle(tmp_path, with_excluded_candidate=True)
    dataset = _dataset(tmp_path)
    campaign = _campaign_config(tmp_path, dataset=dataset, entries=entries)

    publish_response_window_labels(observation_bundle_dir=observation_bundle, dataset_root=dataset)

    with pytest.raises(ResponseWindowLabelPromotionError, match=message):
        promotion_publication.verify_campaign_binding(
            dataset,
            relative_dir=promotion_contracts.confined_relative_directory(DEFAULT_OUTPUT_DIRECTORY),
            expected_width=8,
            campaign_config_path=campaign,
        )

    assert (dataset / DEFAULT_OUTPUT_DIRECTORY).is_dir()


def test_study_publication_does_not_require_an_opal_campaign_binding(tmp_path: Path) -> None:
    observation_bundle = _observation_bundle(tmp_path, with_excluded_candidate=True)
    dataset = _dataset(tmp_path)

    result = publish_response_window_labels(
        observation_bundle_dir=observation_bundle,
        dataset_root=dataset,
    )

    assert result.output_directory.is_dir()


def test_published_bundle_verification_rejects_later_campaign_exclusion_drift(tmp_path: Path) -> None:
    observation_bundle = _observation_bundle(tmp_path, with_excluded_candidate=True)
    dataset = _dataset(tmp_path)
    campaign = _campaign_config(
        tmp_path,
        dataset=dataset,
        entries=[{"candidate_id": "candidate-b", "reason": "nonexact_primary_component"}],
    )
    publish_response_window_labels(
        observation_bundle_dir=observation_bundle,
        dataset_root=dataset,
    )
    payload = yaml.safe_load(campaign.read_text(encoding="utf-8"))
    payload["candidate_eligibility"]["rules"][0]["params"]["entries"][0]["reason"] = "repeat_excluded_noncomparable"
    campaign.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ResponseWindowLabelPromotionError, match="reason mismatch"):
        promotion_publication.verify_campaign_binding(
            dataset,
            relative_dir=promotion_contracts.confined_relative_directory(DEFAULT_OUTPUT_DIRECTORY),
            expected_width=8,
            campaign_config_path=campaign,
        )


def test_explicit_campaign_binding_reverifies_the_configured_x_column(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    publish_response_window_labels(
        observation_bundle_dir=_observation_bundle(tmp_path),
        dataset_root=dataset,
    )
    campaign = _campaign_config(tmp_path, dataset=dataset, entries=[])
    payload = yaml.safe_load(campaign.read_text(encoding="utf-8"))
    payload["data"]["x_column_name"] = "missing_X"
    campaign.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ResponseWindowLabelPromotionError, match="configured candidate/X columns"):
        promotion_publication.verify_campaign_binding(
            dataset,
            relative_dir=promotion_contracts.confined_relative_directory(DEFAULT_OUTPUT_DIRECTORY),
            expected_width=8,
            campaign_config_path=campaign,
        )


def test_publisher_rejects_excluded_candidate_absent_from_candidate_records(tmp_path: Path) -> None:
    observation_bundle = _observation_bundle(tmp_path, with_excluded_candidate=True)
    dataset = _dataset(tmp_path)
    records_path = dataset / "records.parquet"
    records = pd.read_parquet(records_path)
    records.loc[records["id"].ne("candidate-b")].to_parquet(records_path, index=False)
    with pytest.raises(ResponseWindowLabelPromotionError, match="absent from OPAL candidate records"):
        publish_response_window_labels(
            observation_bundle_dir=observation_bundle,
            dataset_root=dataset,
        )

    assert not (dataset / DEFAULT_OUTPUT_DIRECTORY).exists()


def test_publisher_rejects_noncanonical_excluded_candidate_record_id(tmp_path: Path) -> None:
    observation_bundle = _observation_bundle(tmp_path, with_excluded_candidate=True)
    dataset = _dataset(tmp_path)
    records_path = dataset / "records.parquet"
    records = pd.read_parquet(records_path)
    records.loc[records["id"].eq("candidate-b"), "id"] = " candidate-b "
    records.to_parquet(records_path, index=False)
    with pytest.raises(ResponseWindowLabelPromotionError, match="canonical non-empty candidate IDs"):
        publish_response_window_labels(
            observation_bundle_dir=observation_bundle,
            dataset_root=dataset,
        )

    assert not (dataset / DEFAULT_OUTPUT_DIRECTORY).exists()


def test_published_labels_fail_when_candidate_sequence_or_x_snapshot_changes(tmp_path: Path) -> None:
    observation_bundle = _observation_bundle(tmp_path)
    dataset = _dataset(tmp_path)
    result = publish_response_window_labels(
        observation_bundle_dir=observation_bundle,
        dataset_root=dataset,
    )
    records = pd.read_parquet(dataset / "records.parquet")
    records.loc[records["id"].eq("candidate-a"), "sequence"] = "TGCA"
    records.to_parquet(dataset / "records.parquet", index=False)
    binding = ObservedLabelPromotionBinding(
        dataset_root=dataset,
        manifest_path=result.promotion_manifest_path.relative_to(dataset).as_posix(),
        label_path=result.label_path.relative_to(dataset).as_posix(),
        campaign_slug="secg_msrb_greedy",
        study_id="stress_ethanol_cipro_growth",
        y_space="reader_response_window_vector_v1",
    )

    with pytest.raises(ValueError, match="candidate artifact SHA-256"):
        verify_observed_label_snapshot(binding, expected_y_width=8)


def test_candidate_sequence_mismatch_fails_before_output_mutation(tmp_path: Path) -> None:
    observation_bundle = _observation_bundle(tmp_path)
    dataset = _dataset(tmp_path, sequence="TGCA")

    with pytest.raises(ResponseWindowLabelPromotionError, match="sequence digests disagree"):
        publish_response_window_labels(
            observation_bundle_dir=observation_bundle,
            dataset_root=dataset,
        )

    assert not (dataset / DEFAULT_OUTPUT_DIRECTORY).exists()


def test_publisher_rejects_unconfined_output_directory(tmp_path: Path) -> None:
    observation_bundle = _observation_bundle(tmp_path)
    dataset = _dataset(tmp_path)

    with pytest.raises(ResponseWindowLabelPromotionError, match="confined relative"):
        publish_response_window_labels(
            observation_bundle_dir=observation_bundle,
            dataset_root=dataset,
            output_relative_directory="../outside",
        )


def test_publisher_is_create_only_and_preserves_existing_promotion(tmp_path: Path) -> None:
    observation_bundle = _observation_bundle(tmp_path)
    dataset = _dataset(tmp_path)
    first = publish_response_window_labels(
        observation_bundle_dir=observation_bundle,
        dataset_root=dataset,
    )
    before = {
        path.name: path.read_bytes()
        for path in [first.label_path, first.study_provenance_path, first.promotion_manifest_path]
    }

    assert "overwrite" not in signature(publish_response_window_labels).parameters
    with pytest.raises(ResponseWindowLabelPromotionError, match="already exists"):
        publish_response_window_labels(
            observation_bundle_dir=observation_bundle,
            dataset_root=dataset,
        )

    assert {
        path.name: path.read_bytes()
        for path in [first.label_path, first.study_provenance_path, first.promotion_manifest_path]
    } == before


def test_publisher_rejects_observation_drift_during_read(monkeypatch, tmp_path: Path) -> None:
    observation_bundle = _observation_bundle(tmp_path)
    dataset = _dataset(tmp_path)
    original_verify = publisher_module.verify_response_window_observations
    call_count = 0

    def verify_then_drift(path):
        nonlocal call_count
        result = original_verify(path)
        call_count += 1
        if call_count == 1:
            observations = pd.read_parquet(result.observations_parquet)
            observations.loc[0, "r00"] = 99.0
            observations.to_parquet(result.observations_parquet, index=False)
        return result

    monkeypatch.setattr(publisher_module, "verify_response_window_observations", verify_then_drift)

    with pytest.raises(ResponseWindowLabelPromotionError, match="observation bundle drift"):
        publish_response_window_labels(
            observation_bundle_dir=observation_bundle,
            dataset_root=dataset,
        )

    assert not (dataset / DEFAULT_OUTPUT_DIRECTORY).exists()


def test_later_round_cannot_start_a_lineage_without_a_parent(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)

    with pytest.raises(ResponseWindowLabelPromotionError, match="genesis requires no prior.*round 0"):
        publish_response_window_labels(
            observation_bundle_dir=_observation_bundle(tmp_path, observed_round=1, batch_id="batch_1"),
            dataset_root=dataset,
        )

    assert not (dataset / DEFAULT_OUTPUT_DIRECTORY).exists()
    assert not (dataset / "_opal" / promotion_lineage.LINEAGE_HEAD_FILENAME).exists()


def test_authoritative_lineage_head_rejects_a_stale_parent_fork(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    first = publish_response_window_labels(
        observation_bundle_dir=_observation_bundle(tmp_path / "first", observed_round=0, batch_id="batch_0"),
        dataset_root=dataset,
        output_relative_directory="_opal/response_window_labels_round0_v4",
    )
    publish_response_window_labels(
        observation_bundle_dir=_observation_bundle(tmp_path / "second", observed_round=1, batch_id="batch_1"),
        dataset_root=dataset,
        output_relative_directory="_opal/response_window_labels_round1_v4",
        prior_promotion_manifest_path=first.promotion_manifest_path,
    )

    with pytest.raises(ResponseWindowLabelPromotionError, match="prior promotion is stale"):
        publish_response_window_labels(
            observation_bundle_dir=_observation_bundle(tmp_path / "fork", observed_round=2, batch_id="batch_2"),
            dataset_root=dataset,
            output_relative_directory="_opal/response_window_labels_stale_fork_v4",
            prior_promotion_manifest_path=first.promotion_manifest_path,
        )

    assert not (dataset / "_opal/response_window_labels_stale_fork_v4").exists()


def test_three_rounds_may_remeasure_one_candidate_with_new_display_labels(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    prior = None
    labels = ["Candidate A", "A, reviewed", "Promoter A"]
    result = None
    for observed_round, display_label in enumerate(labels):
        result = publish_response_window_labels(
            observation_bundle_dir=_observation_bundle(
                tmp_path / f"round-{observed_round}",
                display_label=display_label,
                observed_round=observed_round,
                batch_id=f"batch_{observed_round}",
            ),
            dataset_root=dataset,
            output_relative_directory=f"_opal/response_window_labels_round{observed_round}_v4",
            prior_promotion_manifest_path=None if prior is None else prior.promotion_manifest_path,
        )
        prior = result

    assert result is not None
    frame = pd.read_parquet(result.label_path)
    assert frame["display_label"].tolist() == labels
    assert result.label_event_count == 3
    assert result.unique_candidate_count == 1


def test_deep_verification_rejects_digest_rebound_provenance_claim_tamper(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    result = publish_response_window_labels(
        observation_bundle_dir=_observation_bundle(tmp_path),
        dataset_root=dataset,
    )
    provenance = json.loads(result.study_provenance_path.read_text(encoding="utf-8"))
    provenance["label_contract"]["label_event_count"] = 99
    result.study_provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    promotion = json.loads(result.promotion_manifest_path.read_text(encoding="utf-8"))
    promotion["study_provenance"]["sha256"] = hashlib.sha256(result.study_provenance_path.read_bytes()).hexdigest()
    result.promotion_manifest_path.write_text(
        json.dumps(promotion, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ResponseWindowLabelPromotionError, match="label-contract provenance disagrees"):
        promotion_publication.verify_label_bundle(
            dataset,
            relative_dir=promotion_contracts.confined_relative_directory(DEFAULT_OUTPUT_DIRECTORY),
            expected_width=8,
        )


def test_copied_source_observation_manifest_is_digest_verified(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    result = publish_response_window_labels(
        observation_bundle_dir=_observation_bundle(tmp_path),
        dataset_root=dataset,
    )
    source = result.output_directory / promotion_contracts.SOURCE_OBSERVATION_MANIFEST_FILENAME
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["policy"]["policy_id"] = "tampered-policy"
    source.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ResponseWindowLabelPromotionError, match="copied source observation manifest digest"):
        promotion_publication.verify_label_bundle(
            dataset,
            relative_dir=promotion_contracts.confined_relative_directory(DEFAULT_OUTPUT_DIRECTORY),
            expected_width=8,
        )


def test_final_verification_failure_removes_newly_renamed_output(monkeypatch, tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    original_verify = promotion_materialization.verify_label_bundle

    def fail_only_after_rename(bundle_root, **kwargs):
        if Path(bundle_root).resolve() == dataset.resolve():
            raise ResponseWindowLabelPromotionError("forced post-rename failure")
        return original_verify(bundle_root, **kwargs)

    monkeypatch.setattr(promotion_materialization, "verify_label_bundle", fail_only_after_rename)

    with pytest.raises(ResponseWindowLabelPromotionError, match="forced post-rename failure"):
        publish_response_window_labels(
            observation_bundle_dir=_observation_bundle(tmp_path),
            dataset_root=dataset,
        )

    assert not (dataset / DEFAULT_OUTPUT_DIRECTORY).exists()
    assert not (dataset / "_opal" / promotion_lineage.LINEAGE_HEAD_FILENAME).exists()


def test_lineage_head_inventory_is_checked_against_verified_labels(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    publish_response_window_labels(
        observation_bundle_dir=_observation_bundle(tmp_path),
        dataset_root=dataset,
    )
    head_path = dataset / "_opal" / promotion_lineage.LINEAGE_HEAD_FILENAME
    head = json.loads(head_path.read_text(encoding="utf-8"))
    head["label_event_count"] = 2
    head_path.write_text(json.dumps(head, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ResponseWindowLabelPromotionError, match="inventory disagrees"):
        promotion_lineage.load_lineage_head(dataset)


def _observation_bundle(
    tmp_path: Path,
    *,
    with_excluded_candidate: bool = False,
    candidate_id: str = "candidate-a",
    display_label: str = "Candidate A",
    sequence: str = "ACGT",
    observed_round: int | None = None,
    batch_id: str | None = None,
    excluded_reason: str = "nonexact_primary_component",
    excluded_candidate_id: str = "candidate-b",
) -> Path:
    evidence = _evidence(tmp_path / "evidence")
    evidence = _retarget_evidence(
        evidence,
        candidate_id=candidate_id,
        display_label=display_label,
        sequence=sequence,
        observed_round=observed_round,
        batch_id=batch_id,
    )
    if with_excluded_candidate:
        excluded = evidence.preview.contributions.iloc[0].copy()
        excluded["candidate_id"] = excluded_candidate_id
        excluded["design_id"] = "design-b"
        excluded["reader_experiment_id"] = "experiment-b"
        excluded["label_source_reader_experiment_id"] = "experiment-b"
        excluded["included_in_label"] = False
        excluded["label_exclusion_reason"] = excluded_reason
        excluded["r01_bound_kind"] = "lower"
        excluded["r01_has_instrument_overflow"] = True
        evidence = replace(
            evidence,
            preview=replace(
                evidence.preview,
                contributions=pd.concat(
                    [evidence.preview.contributions, excluded.to_frame().T],
                    ignore_index=True,
                ),
            ),
        )
    output = tmp_path / "observation-bundle"
    materialize_response_window_observations(
        evidence,
        out_dir=output,
        allowed_output_root=tmp_path,
    )
    return output


def _retarget_evidence(
    evidence,
    *,
    candidate_id: str,
    display_label: str,
    sequence: str,
    observed_round: int | None,
    batch_id: str | None,
):
    preview = evidence.preview
    frames = {}
    for field in (
        "observations",
        "contributions",
        "bootstrap_draws",
        "uncertainty",
        "repeat_diagnostics",
        "reduction_sensitivity",
        "event_time_sensitivity",
    ):
        frame = getattr(preview, field).copy()
        if "candidate_id" in frame.columns:
            frame["candidate_id"] = frame["candidate_id"].replace("candidate-a", candidate_id)
        frames[field] = frame
    frames["observations"].loc[:, "display_label"] = display_label
    frames["observations"].loc[:, "sequence_sha256"] = hashlib.sha256(sequence.encode("utf-8")).hexdigest()
    pd.DataFrame({"candidate_id": [candidate_id]}).to_parquet(evidence.candidate_bindings_path, index=False)
    binding_manifest = json.loads(evidence.candidate_bindings_manifest_path.read_text(encoding="utf-8"))
    binding_manifest["record"]["sha256"] = hashlib.sha256(evidence.candidate_bindings_path.read_bytes()).hexdigest()
    evidence.candidate_bindings_manifest_path.write_text(
        json.dumps(binding_manifest),
        encoding="utf-8",
    )
    binding_sha = hashlib.sha256(evidence.candidate_bindings_manifest_path.read_bytes()).hexdigest()
    policy = replace(
        evidence.policy,
        observed_round=evidence.policy.observed_round if observed_round is None else observed_round,
        batch_id=evidence.policy.batch_id if batch_id is None else batch_id,
        candidate_bindings_sha256=binding_sha,
    )
    return replace(
        evidence,
        policy=policy,
        preview=replace(preview, **frames),
        candidate_bindings_manifest_sha256=binding_sha,
    )


def _dataset(tmp_path: Path, *, sequence: str = "ACGT") -> Path:
    root = tmp_path / "dataset"
    root.mkdir()
    records = pd.DataFrame(
        {
            "id": ["candidate-a", "candidate-b", "candidate-unmeasured"],
            "sequence": [sequence, "CCCC", "AAAA"],
            "bio_type": ["dna", "dna", "dna"],
            "alphabet": ["dna_4", "dna_4", "dna_4"],
        }
    )
    table = pa.Table.from_pandas(records, preserve_index=False).append_column(
        "X",
        pa.array([[0.1], [0.2], [0.3]], type=pa.list_(pa.float32(), list_size=1)),
    )
    pq.write_table(table, root / "records.parquet")
    return root


def _run_opal(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", "from dnadesign.opal import main; main()", *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _campaign_config(
    tmp_path: Path,
    *,
    dataset: Path,
    entries: list[dict[str, str]],
    label_directory: str = DEFAULT_OUTPUT_DIRECTORY,
) -> Path:
    source = Path("src/dnadesign/opal/campaigns/secg_msrb_greedy/configs/campaign.yaml")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["campaign"]["workdir"] = str(tmp_path / "campaign-workdir")
    payload["ownership"]["dataset_id"] = dataset.name
    payload["data"]["location"] = {
        "kind": "usr",
        "path": str(dataset.parent),
        "dataset": dataset.name,
    }
    payload["data"]["x_column_name"] = "X"
    payload["labels"]["source"]["dataset"] = dataset.name
    payload["labels"]["source"]["path"] = f"{label_directory}/observed_labels.parquet"
    payload["labels"]["source"]["manifest_path"] = f"{label_directory}/promotion.manifest.json"
    payload["candidate_eligibility"] = {"rules": []}
    if entries:
        payload["candidate_eligibility"]["rules"].append(
            {
                "name": "candidate_id_exclusion",
                "params": {
                    "exclusion_set_id": "stress_response_window_observation_dispositions_v1",
                    "entries": entries,
                    "min_remaining_candidates": 1,
                },
            }
        )
    campaign = tmp_path / "campaign" / "configs" / "campaign.yaml"
    campaign.parent.mkdir(parents=True, exist_ok=True)
    campaign.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return campaign


def _runnable_campaign_config(
    tmp_path: Path,
    *,
    dataset: Path,
    label_directory: str,
) -> Path:
    campaign = _campaign_config(
        tmp_path,
        dataset=dataset,
        entries=[],
        label_directory=label_directory,
    )
    payload = yaml.safe_load(campaign.read_text(encoding="utf-8"))
    payload["plot_config"] = str(Path("src/dnadesign/opal/campaigns/secg_msrb_greedy/configs/plots.yaml").resolve())
    payload["model"]["params"].update(
        {
            "n_estimators": 5,
            "oob_score": False,
            "n_jobs": 1,
            "emit_feature_importance": False,
        }
    )
    payload["selection_views"] = [payload["selection_views"][0]]
    payload["selection_views"][0]["selection"]["params"]["top_k"] = 1
    payload["selection_batch"] = {
        "deduplicate_by": "sequence",
        "expected_unique_count": 1,
    }
    campaign.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return campaign
