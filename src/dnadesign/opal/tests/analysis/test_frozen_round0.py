"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/analysis/test_frozen_round0.py

End-to-end contract tests for OPAL frozen round-zero replay scoring.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dnadesign.opal.src.analysis.learning_loop_baselines import frozen_round0_scores
from dnadesign.opal.tests._cli_helpers import write_campaign_yaml


def _write_scalar_campaign(tmp_path: Path) -> Path:
    records_path = tmp_path / "records.parquet"
    records = pd.DataFrame(
        {
            "id": ["seed-a", "seed-b", "candidate-a", "candidate-b"],
            "sequence": ["AAAA", "CCCC", "GGGG", "TTTT"],
            "bio_type": ["dna"] * 4,
            "alphabet": ["dna_4"] * 4,
            "X": [[0.0, 0.0], [0.2, 0.2], [0.8, 0.8], [1.0, 1.0]],
            "opal__demo__label_hist": [
                [
                    {
                        "kind": "label",
                        "observed_round": 0,
                        "y_obs": {"value": [0.0], "dtype": "vector", "schema": {"length": 1}},
                    }
                ],
                [
                    {
                        "kind": "label",
                        "observed_round": 0,
                        "y_obs": {"value": [1.0], "dtype": "vector", "schema": {"length": 1}},
                    }
                ],
                [],
                [],
            ],
            "Y": [None] * 4,
        }
    )
    records.to_parquet(records_path, index=False)

    config_path = tmp_path / "campaign.yaml"
    write_campaign_yaml(
        config_path,
        workdir=tmp_path / "campaign",
        records_path=records_path,
        transforms_y_name="scalar_from_table_v1",
        objective_name="scalar_identity_v1",
        objective_params={},
        y_expected_length=1,
        selection_params={"exclude_already_labeled": True},
    )
    return config_path


def test_frozen_round0_scores_replays_real_scalar_campaign(tmp_path: Path) -> None:
    config_path = _write_scalar_campaign(tmp_path)

    scores, seed_ids = frozen_round0_scores(config_path, selection_view_id="primary")

    assert seed_ids == ["seed-a", "seed-b"]
    assert scores["id"].tolist() == ["candidate-a", "candidate-b"]
    assert scores["id"].is_unique
    assert np.isfinite(scores["score"]).all()
    assert scores["run_id"].nunique() == 1
    assert scores["campaign_config_path"].tolist() == [str(config_path)] * 2
