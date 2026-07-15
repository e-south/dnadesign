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
from inspect import signature
from pathlib import Path

import pandas as pd
import pytest

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
    assert provenance["observation_bundle"]["schema_id"].endswith("response_window_observations.v1")


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


def _observation_bundle(tmp_path: Path) -> Path:
    evidence = _evidence(tmp_path / "evidence")
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
    pd.DataFrame(
        {
            "id": ["candidate-a", "candidate-unmeasured"],
            "sequence": [sequence, "AAAA"],
        }
    ).to_parquet(root / "records.parquet", index=False)
    return root
