"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/study/test_study_spec_strict_validation.py

Validate strict Study spec contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dnadesign.cruncher.study.schema_models import StudyRoot


def _base_payload() -> dict:
    return {
        "study": {
            "schema_version": 3,
            "name": "diversity_length_diag",
            "base_config": "./config.yaml",
            "target": {"kind": "regulator_set", "set_index": 1},
            "execution": {
                "parallelism": 1,
                "on_trial_error": "continue",
                "exit_code_policy": "nonzero_if_any_error",
                "summarize_after_run": True,
            },
            "artifacts": {"trial_output_profile": "minimal"},
            "replicates": {"seed_path": "sample.seed", "seeds": [1, 2]},
            "trials": [
                {
                    "id": "L16",
                    "factors": {
                        "sample.sequence_length": 16,
                        "sample.elites.select.diversity": 0.15,
                    },
                }
            ],
            "replays": {
                "mmr_sweep": {
                    "enabled": True,
                    "pool_size_values": ["auto"],
                    "diversity_values": [0.0, 0.25, 0.5],
                }
            },
        }
    }


def test_spec_rejects_unknown_top_level_key() -> None:
    payload = _base_payload()
    payload["study"]["extra_key"] = {"x": 1}
    with pytest.raises(ValidationError):
        StudyRoot.model_validate(payload)


def test_spec_rejects_duplicate_trial_ids() -> None:
    payload = _base_payload()
    payload["study"]["trials"] = [
        {"id": "L16", "factors": {"sample.sequence_length": 16}},
        {"id": "L16", "factors": {"sample.sequence_length": 20}},
    ]
    with pytest.raises(ValidationError, match="trial ids must be unique"):
        StudyRoot.model_validate(payload)


def test_spec_rejects_legacy_output_root_key() -> None:
    payload = _base_payload()
    payload["study"]["artifacts"]["output_root"] = "outputs/studies"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        StudyRoot.model_validate(payload)


def test_spec_rejects_invalid_trial_id() -> None:
    payload = _base_payload()
    payload["study"]["trials"][0]["id"] = "bad id with spaces"
    with pytest.raises(ValidationError, match="slug-safe"):
        StudyRoot.model_validate(payload)


def test_spec_accepts_trial_grids_without_explicit_trials() -> None:
    payload = _base_payload()
    payload["study"]["trials"] = []
    payload["study"]["trial_grids"] = [
        {
            "id_prefix": "G",
            "factors": {
                "sample.sequence_length": [16, 20],
                "sample.elites.select.diversity": [0.0, 0.5],
            },
        }
    ]
    model = StudyRoot.model_validate(payload)
    assert model.study.trials == []
    assert len(model.study.trial_grids) == 1


def test_spec_rejects_when_trials_and_trial_grids_are_both_empty() -> None:
    payload = _base_payload()
    payload["study"]["trials"] = []
    payload["study"]["trial_grids"] = []
    with pytest.raises(ValidationError, match="must define at least one trial"):
        StudyRoot.model_validate(payload)


def test_spec_defaults_trial_output_profile_to_minimal() -> None:
    payload = _base_payload()
    payload["study"].pop("artifacts")
    model = StudyRoot.model_validate(payload)
    assert model.study.artifacts.trial_output_profile == "minimal"


def test_spec_accepts_parallelism_above_one() -> None:
    payload = _base_payload()
    payload["study"]["execution"]["parallelism"] = 2
    model = StudyRoot.model_validate(payload)
    assert model.study.execution.parallelism == 2


def test_spec_defaults_parallelism_to_six_when_omitted() -> None:
    payload = _base_payload()
    execution = payload["study"]["execution"]
    assert isinstance(execution, dict)
    execution.pop("parallelism")
    model = StudyRoot.model_validate(payload)
    assert model.study.execution.parallelism == 6


def test_spec_rejects_non_sweep_factor_key() -> None:
    payload = _base_payload()
    payload["study"]["trials"][0]["factors"]["sample.budget.tune"] = 1500
    with pytest.raises(ValidationError, match="study factor key is not supported"):
        StudyRoot.model_validate(payload)
