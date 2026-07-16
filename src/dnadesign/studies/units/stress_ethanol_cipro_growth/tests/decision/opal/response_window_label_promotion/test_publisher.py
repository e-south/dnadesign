"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/response_window_label_promotion/test_publisher.py

End-to-end contract tests for study-owned OPAL label promotion.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from dataclasses import replace
from inspect import signature
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from dnadesign.opal import (
    ObservedLabelPromotionBinding,
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
        campaign_config_path=None,
    )

    labels = pd.read_parquet(result.label_path)
    assert labels.columns.tolist() == ["id", "observed_round", "batch_id", "y_space", "y_obs"]
    assert labels["id"].tolist() == ["candidate-a"]
    assert labels["y_obs"].map(lambda value: getattr(value, "ndim", 1)).tolist() == [1]
    assert labels["y_obs"].map(len).tolist() == [8]
    assert result.output_directory.relative_to(dataset).as_posix() == DEFAULT_OUTPUT_DIRECTORY

    binding = ObservedLabelPromotionBinding(
        dataset_root=dataset,
        manifest_path=f"{DEFAULT_OUTPUT_DIRECTORY}/promotion.manifest.json",
        label_path=f"{DEFAULT_OUTPUT_DIRECTORY}/observed_labels.parquet",
        campaign_slug="secg_rmf_greedy",
        study_id="stress_ethanol_cipro_growth",
        y_space="reader_response_window_vector_v1",
    )
    verified = verify_observed_label_snapshot(binding, expected_y_width=8)
    assert verified.promotion.row_count == 1
    assert verified.promotion.candidate_path == (dataset / "records.parquet").resolve()
    provenance = json.loads(result.study_provenance_path.read_text(encoding="utf-8"))
    assert provenance["observation_bundle"]["schema_id"].endswith("response_window_observations.v2")


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
        campaign_config_path=None,
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
        campaign_config_path=campaign,
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
def test_publisher_rejects_campaign_exclusion_drift(
    tmp_path: Path,
    entries: list[dict[str, str]],
    message: str,
) -> None:
    observation_bundle = _observation_bundle(tmp_path, with_excluded_candidate=True)
    dataset = _dataset(tmp_path)
    campaign = _campaign_config(tmp_path, dataset=dataset, entries=entries)

    with pytest.raises(ResponseWindowLabelPromotionError, match=message):
        publish_response_window_labels(
            observation_bundle_dir=observation_bundle,
            dataset_root=dataset,
            campaign_config_path=campaign,
        )

    assert not (dataset / DEFAULT_OUTPUT_DIRECTORY).exists()


def test_publisher_requires_campaign_binding_for_nonempty_candidate_exclusions(tmp_path: Path) -> None:
    observation_bundle = _observation_bundle(tmp_path, with_excluded_candidate=True)
    dataset = _dataset(tmp_path)

    with pytest.raises(ResponseWindowLabelPromotionError, match="requires a campaign config"):
        publish_response_window_labels(
            observation_bundle_dir=observation_bundle,
            dataset_root=dataset,
            campaign_config_path=None,
        )

    assert not (dataset / DEFAULT_OUTPUT_DIRECTORY).exists()


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
        campaign_config_path=campaign,
    )
    payload = yaml.safe_load(campaign.read_text(encoding="utf-8"))
    payload["candidate_eligibility"]["rules"][0]["params"]["entries"][0]["reason"] = "repeat_excluded_noncomparable"
    campaign.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ResponseWindowLabelPromotionError, match="reason mismatch"):
        promotion_publication.verify_label_bundle(
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
    campaign = _campaign_config(
        tmp_path,
        dataset=dataset,
        entries=[{"candidate_id": "candidate-b", "reason": "nonexact_primary_component"}],
    )

    with pytest.raises(ResponseWindowLabelPromotionError, match="absent from OPAL candidate records"):
        publish_response_window_labels(
            observation_bundle_dir=observation_bundle,
            dataset_root=dataset,
            campaign_config_path=campaign,
        )

    assert not (dataset / DEFAULT_OUTPUT_DIRECTORY).exists()


def test_publisher_rejects_noncanonical_excluded_candidate_record_id(tmp_path: Path) -> None:
    observation_bundle = _observation_bundle(tmp_path, with_excluded_candidate=True)
    dataset = _dataset(tmp_path)
    records_path = dataset / "records.parquet"
    records = pd.read_parquet(records_path)
    records.loc[records["id"].eq("candidate-b"), "id"] = " candidate-b "
    records.to_parquet(records_path, index=False)
    campaign = _campaign_config(
        tmp_path,
        dataset=dataset,
        entries=[{"candidate_id": "candidate-b", "reason": "nonexact_primary_component"}],
    )

    with pytest.raises(ResponseWindowLabelPromotionError, match="canonical non-empty candidate IDs"):
        publish_response_window_labels(
            observation_bundle_dir=observation_bundle,
            dataset_root=dataset,
            campaign_config_path=campaign,
        )

    assert not (dataset / DEFAULT_OUTPUT_DIRECTORY).exists()


def test_published_labels_fail_when_candidate_sequence_or_x_snapshot_changes(tmp_path: Path) -> None:
    observation_bundle = _observation_bundle(tmp_path)
    dataset = _dataset(tmp_path)
    result = publish_response_window_labels(
        observation_bundle_dir=observation_bundle,
        dataset_root=dataset,
        campaign_config_path=None,
    )
    records = pd.read_parquet(dataset / "records.parquet")
    records.loc[records["id"].eq("candidate-a"), "sequence"] = "TGCA"
    records.to_parquet(dataset / "records.parquet", index=False)
    binding = ObservedLabelPromotionBinding(
        dataset_root=dataset,
        manifest_path=result.promotion_manifest_path.relative_to(dataset).as_posix(),
        label_path=result.label_path.relative_to(dataset).as_posix(),
        campaign_slug="secg_rmf_greedy",
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
            campaign_config_path=None,
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
            campaign_config_path=None,
        )


def test_publisher_is_create_only_and_preserves_existing_promotion(tmp_path: Path) -> None:
    observation_bundle = _observation_bundle(tmp_path)
    dataset = _dataset(tmp_path)
    first = publish_response_window_labels(
        observation_bundle_dir=observation_bundle,
        dataset_root=dataset,
        campaign_config_path=None,
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
            campaign_config_path=None,
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
            campaign_config_path=None,
        )

    assert not (dataset / DEFAULT_OUTPUT_DIRECTORY).exists()


def _observation_bundle(tmp_path: Path, *, with_excluded_candidate: bool = False) -> Path:
    evidence = _evidence(tmp_path / "evidence")
    if with_excluded_candidate:
        excluded = evidence.preview.contributions.iloc[0].copy()
        excluded["candidate_id"] = "candidate-b"
        excluded["design_id"] = "design-b"
        excluded["reader_experiment_id"] = "experiment-b"
        excluded["label_source_reader_experiment_id"] = "experiment-b"
        excluded["included_in_label"] = False
        excluded["label_exclusion_reason"] = "nonexact_primary_component"
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


def _campaign_config(
    tmp_path: Path,
    *,
    dataset: Path,
    entries: list[dict[str, str]],
) -> Path:
    source = Path("src/dnadesign/opal/campaigns/secg_rmf_greedy/configs/campaign.yaml")
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
