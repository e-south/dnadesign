"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/response_metastudy/test_label_truth.py

Configured observed-label truth tests for the response metastudy.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from dnadesign.opal import ObservedLabelVerificationError
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.runtime import (
    label_truth,
    manifest,
)


def test_missing_configured_promotion_manifest_is_not_ready(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(label_truth, "load_config", lambda _path: config)

    state = label_truth.resolve_configured_label_truth(tmp_path / "campaign.yaml")

    assert state.state == "not_ready"
    assert state.label_source_state == "not_verified"
    assert state.observed_label_promotion_manifest is None
    assert state.ready is False


def test_invalid_configured_promotion_manifest_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    manifest_path = _manifest_path(tmp_path)
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(label_truth, "load_config", lambda _path: config)

    def reject(*_args, **_kwargs):
        raise ObservedLabelVerificationError("invalid promotion")

    monkeypatch.setattr(label_truth, "verify_observed_label_snapshot", reject)

    with pytest.raises(ObservedLabelVerificationError, match="invalid promotion"):
        label_truth.resolve_configured_label_truth(tmp_path / "campaign.yaml")


def test_verified_configured_promotion_manifest_is_promoted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    manifest_path = _manifest_path(tmp_path)
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(label_truth, "load_config", lambda _path: config)
    monkeypatch.setattr(
        label_truth,
        "verify_observed_label_snapshot",
        lambda *_args, **_kwargs: SimpleNamespace(
            promotion=SimpleNamespace(manifest_sha256="a" * 64),
            labels=pd.DataFrame({"id": ["candidate-a", "candidate-b"]}),
        ),
    )

    state = label_truth.resolve_configured_label_truth(tmp_path / "campaign.yaml")

    assert state.state == "promoted"
    assert state.label_source_state == "verified"
    assert state.observed_label_promotion_manifest == {
        "path": "_opal/labels/promotion.manifest.json",
        "sha256": "a" * 64,
    }
    assert state.candidate_ids == ("candidate-a", "candidate-b")
    assert state.ready is True


def test_manifest_label_truth_keeps_screen_selection_non_authoritative() -> None:
    state = label_truth.LabelTruthState(
        state="promoted",
        label_source_state="verified",
        observed_label_promotion_manifest={"path": "promotion.manifest.json", "sha256": "a" * 64},
    )

    record = manifest.build_label_truth_record(
        state,
        screen_source_scope="model_screen_only",
        screen_source_label_truth_role="none",
    )

    assert record == {
        "state": "promoted",
        "source": "stress_ethanol_cipro_growth.response_window_observations",
        "screen_source_scope": "model_screen_only",
        "screen_source_label_truth_role": "none",
        "label_source_state": "verified",
        "observed_label_promotion_manifest": {
            "path": "promotion.manifest.json",
            "sha256": "a" * 64,
        },
    }


def _manifest_path(tmp_path: Path) -> Path:
    return tmp_path / "candidate-dataset/_opal/labels/promotion.manifest.json"


def _config(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        campaign=SimpleNamespace(slug="secg_rmf_greedy"),
        ownership=SimpleNamespace(owner_scope="study_campaign", study_id="stress_ethanol_cipro_growth"),
        data=SimpleNamespace(
            location=SimpleNamespace(kind="usr", path=str(tmp_path), dataset="candidate-dataset"),
            y_expected_length=8,
            x_column_name="x",
        ),
        labels=SimpleNamespace(
            source=SimpleNamespace(
                kind="usr_sidecar",
                dataset="candidate-dataset",
                path="_opal/labels/observed_labels.parquet",
                manifest_path="_opal/labels/promotion.manifest.json",
            ),
            y_space="reader_response_window_vector_v1",
            id_column="id",
        ),
        candidate_eligibility=SimpleNamespace(rules=[]),
    )
