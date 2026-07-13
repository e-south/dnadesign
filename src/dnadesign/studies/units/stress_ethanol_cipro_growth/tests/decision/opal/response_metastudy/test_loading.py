"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/response_metastudy/test_loading.py

Input-contract tests for the response metric metastudy runtime.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.core.contracts import (
    SFXI_SOURCE_PROVENANCE,
    MetastudyPaths,
    SfxiSourceProvenance,
    StressTargetView,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.runtime.loading import (
    assert_sfxi_run_contract,
    assert_shared_observed_labels,
    load_label_source_frame,
    load_sfxi_evidence_frame,
    load_stress_campaign_contract,
    load_training_matrix,
)


def test_training_matrix_is_aligned_to_label_order(tmp_path) -> None:
    records_path = tmp_path / "records.parquet"
    pd.DataFrame(
        {
            "id": ["b", "a"],
            "x": [[3.0, 4.0], [1.0, 2.0]],
        }
    ).to_parquet(records_path, index=False)
    labels = pd.DataFrame(
        {
            "id": ["a", "b"],
            "y_obs": [np.arange(8, dtype=float), np.arange(8, dtype=float) + 10.0],
        }
    )

    x, y = load_training_matrix(records_path, x_column="x", labels=labels)

    assert x.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert y[:, 0].tolist() == [0.0, 10.0]


def test_shared_label_contract_rejects_source_drift() -> None:
    first = pd.DataFrame({"id": ["a"], "sequence": ["AAAA"], "y_obs": [np.arange(8, dtype=float)]})
    drifted = first.copy()
    drifted.at[0, "y_obs"] = np.arange(8, dtype=float) + 1.0

    with pytest.raises(ValueError, match="observed label ledgers are not identical"):
        assert_shared_observed_labels((first, drifted))


def test_sfxi_run_contract_rejects_target_mask_drift() -> None:
    run = _aligned_run()
    run["objective__params"]["setpoint_vector"] = np.asarray([0.0, 0.0, 1.0, 1.0])

    with pytest.raises(ValueError, match="setpoint_vector"):
        assert_sfxi_run_contract(run, source=_sfxi_source(), target_view=_target_view())


def test_label_source_frame_aligns_reader_experiments_to_ledger_order(tmp_path) -> None:
    source = _sfxi_source()
    campaign_dir = tmp_path / "campaigns" / source.source_campaign_slug
    input_dir = campaign_dir / "inputs" / "r0"
    input_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "id": ["b", "a"],
            "design_id": ["B", "A"],
            "reader_experiment_id": ["exp-b", "exp-a"],
            "v00": [0.0, 0.0],
            "v10": [0.0, 0.0],
            "v01": [0.0, 0.0],
            "v11": [1.0, 1.0],
            "y00_star": [0.0, 0.0],
            "y10_star": [0.0, 0.0],
            "y01_star": [0.0, 0.0],
            "y11_star": [1.0, 1.0],
        }
    ).to_csv(input_dir / "reader_vec8_batch0.csv", index=False)
    labels = pd.DataFrame(
        {
            "id": ["a", "b"],
            "sequence": ["AAAA", "CCCC"],
            "y_obs": [np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])] * 2,
        }
    )
    paths = MetastudyPaths(
        repo_root=tmp_path,
        reader_bundle_root=tmp_path / "reader-bundle",
        out_dir=tmp_path / "out",
        campaign_root=tmp_path / "campaigns",
    )

    sources = load_label_source_frame(paths, source, labels=labels)

    assert sources["id"].tolist() == ["a", "b"]
    assert sources["reader_experiment_id"].tolist() == ["exp-a", "exp-b"]


def test_label_source_frame_rejects_identity_drift(tmp_path) -> None:
    source = _sfxi_source()
    input_dir = tmp_path / "campaigns" / source.source_campaign_slug / "inputs" / "r0"
    input_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "id": ["other"],
            "design_id": ["D"],
            "reader_experiment_id": ["exp"],
            **{
                column: [0.0] for column in ("v00", "v10", "v01", "v11", "y00_star", "y10_star", "y01_star", "y11_star")
            },
        }
    ).to_csv(input_dir / "reader_vec8_batch0.csv", index=False)
    labels = pd.DataFrame({"id": ["expected"], "sequence": ["AAAA"], "y_obs": [np.zeros(8)]})
    paths = MetastudyPaths(
        repo_root=tmp_path,
        reader_bundle_root=tmp_path / "reader-bundle",
        out_dir=tmp_path / "out",
        campaign_root=tmp_path / "campaigns",
    )

    with pytest.raises(ValueError, match="identity does not match"):
        load_label_source_frame(paths, source, labels=labels)


def test_real_repository_sfxi_sources_load_from_persisted_artifacts(tmp_path) -> None:
    repo_root = next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file())
    paths = MetastudyPaths(
        repo_root=repo_root,
        reader_bundle_root=tmp_path / "reader-bundle",
        out_dir=tmp_path / "out",
        campaign_root=repo_root / "src/dnadesign/opal/campaigns",
    )

    stress_campaign = load_stress_campaign_contract(paths)
    target_views = {target_view.id: target_view for target_view in stress_campaign.target_views}
    sfxi_evidence = tuple(
        load_sfxi_evidence_frame(
            paths,
            source,
            target_view=target_views[source.target_view_id],
            stress_campaign=stress_campaign,
        )
        for source in SFXI_SOURCE_PROVENANCE
    )

    assert tuple(target_views) == ("ethanol", "ciprofloxacin", "and")
    assert tuple(run.source.lifecycle for run in sfxi_evidence) == ("provenance_only",) * 3
    assert tuple(run.target_view.target_mask for run in sfxi_evidence) == (
        (0.0, 1.0, 0.0, 1.0),
        (0.0, 0.0, 1.0, 1.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def _sfxi_source() -> SfxiSourceProvenance:
    return SfxiSourceProvenance(
        source_id="test-sfxi-ethanol",
        source_campaign_slug="test-sfxi-ethanol",
        expected_run_id="r0",
        target_view_id="ethanol",
    )


def _target_view() -> StressTargetView:
    return StressTargetView("ethanol", "Ethanol", (0.0, 1.0, 0.0, 1.0))


def _aligned_run() -> pd.Series:
    objective_params = {
        "setpoint_vector": [0.0, 1.0, 0.0, 1.0],
        "logic_exponent_beta": 1.0,
        "intensity_exponent_gamma": 1.0,
        "intensity_log2_offset_delta": 0.0,
        "scaling": {"percentile": 95, "min_n": 5, "eps": 1.0e-8},
    }
    model_params = {"n_estimators": 10, "random_state": 7}
    y_ops = [{"name": "intensity_median_iqr", "params": {"eps": 1.0e-8}}]
    selection_params = {
        "top_k": 6,
        "score_ref": "sfxi_v1/sfxi",
        "tie_handling": "competition_rank",
        "objective_mode": "maximize",
    }
    run = pd.Series(
        {
            "run_id": "r0",
            "objective__name": "sfxi_v1",
            "objective__params": {**objective_params, "setpoint_vector": np.asarray([0.0, 1.0, 0.0, 1.0])},
            "objective__denom_percentile": 95,
            "model__name": "random_forest",
            "model__params": model_params,
            "training__y_ops": y_ops,
            "selection__name": "top_n",
            "selection__params": selection_params,
            "selection__score_ref": "sfxi_v1/sfxi",
            "selection__objective": "maximize",
            "selection__tie_handling": "competition_rank",
        }
    )
    return run
