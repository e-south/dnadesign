"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/analysis/test_observed_objective_history.py

Contract tests for run-pinned observed objective history projections.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from dnadesign.opal.src.analysis.observed_objective_history import (
    load_observed_objective_history,
    observed_objective_run_contract_sha256,
)
from dnadesign.opal.src.core.utils import OpalError, file_sha256
from dnadesign.opal.src.storage.artifacts import run_scoped_artifact_path

_CALIBRATION = {
    "response_separation_min": 0.0,
    "on_magnitude_min": 0.0,
    "off_magnitude_max": 0.0,
    "response_separation_scale": 1.0,
    "on_magnitude_scale": 1.0,
    "off_magnitude_scale": 1.0,
}


def _run_row(
    outputs_dir: Path,
    *,
    round_k: int,
    run_id: str,
    events: list[dict[str, object]],
    calibration: dict[str, float] | None = None,
    target_mask: list[int] | None = None,
    y_space: str = "reader_response_window_vector_v1",
    score_channel: str = "feasibility_margin",
    source_kind: str = "usr_sidecar",
) -> dict[str, object]:
    labels_path = run_scoped_artifact_path(
        outputs_dir / "rounds" / f"round_{round_k}",
        run_id=run_id,
        artifact_key="labels/observed_events.parquet",
    )
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(events).with_columns(
        pl.lit(run_id).alias("run_id"),
        pl.lit(round_k).alias("as_of_round"),
        pl.lit(y_space).alias("y_space"),
        pl.lit(source_kind).alias("label_source_kind"),
    ).write_parquet(labels_path)
    objective = {
        "selection_view_id": "ethanol",
        "objective_name": "response_magnitude_feasibility_v1",
        "params": {
            "state_ids": ["00", "10", "01", "11"],
            "target_mask": list(target_mask or [0, 1, 0, 1]),
            "calibration": dict(calibration or _CALIBRATION),
        },
        "score_channels": [f"ethanol/{score_channel}"],
        "uncertainty_channels": [],
    }
    selection = {
        "selection_view_id": "ethanol",
        "objective_name": "response_magnitude_feasibility_v1",
        "objective_params": dict(objective["params"]),
        "score_ref": f"ethanol/{score_channel}",
        "objective_mode": "minimize" if score_channel == "off_magnitude_ceiling" else "maximize",
    }
    return {
        "run_id": run_id,
        "as_of_round": round_k,
        "y_ingest__name": "vector_from_table_v1",
        "y_ingest__params": {
            "id_column": "id",
            "sequence_column": "sequence",
            "value_columns": ["r00", "r10", "r01", "r11", "b00", "b10", "b01", "b11"],
        },
        "objective__defs_json": json.dumps([objective]),
        "selection_views__defs_json": json.dumps([selection]),
        "artifacts": {"labels/observed_events.parquet": [file_sha256(labels_path), str(labels_path.resolve())]},
    }


def _event(
    candidate_id: str,
    *,
    observed_round: int,
    batch_id: str | None,
    y_obs: list[float],
) -> dict[str, object]:
    return {
        "id": candidate_id,
        "display_label": candidate_id.upper(),
        "sequence": "ACGT",
        "observed_round": observed_round,
        "batch_id": batch_id,
        "y_obs": y_obs,
    }


def test_campaign_history_derives_a_round_batch_label(tmp_path: Path) -> None:
    outputs_dir = tmp_path / "outputs"
    event = _event(
        "candidate-a",
        observed_round=0,
        batch_id=None,
        y_obs=[0.0, 2.0, 0.0, 2.0, -1.0, 1.0, -1.0, 1.0],
    )
    row = _run_row(
        outputs_dir,
        round_k=0,
        run_id="r0",
        events=[event],
        source_kind="campaign_history",
    )
    ledger_dir = outputs_dir / "ledger"
    ledger_dir.mkdir(parents=True)
    pl.DataFrame([row]).write_parquet(ledger_dir / "runs.parquet")
    digest = observed_objective_run_contract_sha256(
        outputs_dir=outputs_dir,
        selection_view_id="ethanol",
        as_of_round=0,
        run_id="r0",
    )

    history = load_observed_objective_history(
        outputs_dir=outputs_dir,
        selection_view_id="ethanol",
        run_series={
            "schema_version": "opal.observed_objective_run_series.v1",
            "runs": [{"as_of_round": 0, "run_id": "r0", "contract_sha256": digest}],
        },
    )

    assert history.frame["batch_id"].tolist() == ["round-0"]


def test_run_series_requires_an_explicit_contract_digest(tmp_path: Path) -> None:
    with pytest.raises(OpalError, match="contract_sha256"):
        load_observed_objective_history(
            outputs_dir=tmp_path / "outputs",
            selection_view_id="ethanol",
            run_series={
                "schema_version": "opal.observed_objective_run_series.v1",
                "runs": [{"as_of_round": 0, "run_id": "r0"}],
            },
        )


def test_history_scores_each_event_once_across_cumulative_run_snapshots(tmp_path: Path) -> None:
    outputs_dir = tmp_path / "outputs"
    first = _event(
        "candidate-a",
        observed_round=0,
        batch_id="batch-0",
        y_obs=[0.0, 2.0, 0.0, 2.0, -1.0, 1.0, -1.0, 1.0],
    )
    second = _event(
        "candidate-b",
        observed_round=1,
        batch_id="batch-1",
        y_obs=[0.0, 1.0, 0.0, 1.0, -0.5, 0.5, -0.5, 0.5],
    )
    third = _event(
        "candidate-c",
        observed_round=0,
        batch_id="batch-0",
        y_obs=[0.0, 0.0, 0.0, 0.0, -1.0, 1.0, -1.0, 1.0],
    )
    rows = [
        _run_row(outputs_dir, round_k=0, run_id="r0", events=[first, third]),
        _run_row(outputs_dir, round_k=1, run_id="r1", events=[first, third, second]),
    ]
    ledger_dir = outputs_dir / "ledger"
    ledger_dir.mkdir(parents=True)
    pl.DataFrame(rows).write_parquet(ledger_dir / "runs.parquet")
    run_series = {
        "schema_version": "opal.observed_objective_run_series.v1",
        "runs": [
            {
                "as_of_round": round_k,
                "run_id": run_id,
                "contract_sha256": observed_objective_run_contract_sha256(
                    outputs_dir=outputs_dir,
                    selection_view_id="ethanol",
                    as_of_round=round_k,
                    run_id=run_id,
                ),
            }
            for round_k, run_id in [(0, "r0"), (1, "r1")]
        ],
    }

    history = load_observed_objective_history(
        outputs_dir=outputs_dir,
        selection_view_id="ethanol",
        run_series=run_series,
    )

    assert history.objective_name == "response_magnitude_feasibility_v1"
    assert history.score_ref == "ethanol/feasibility_margin"
    assert history.objective_mode == "maximize"
    assert history.y_space == "reader_response_window_vector_v1"
    assert history.frame["id"].tolist() == ["candidate-a", "candidate-c", "candidate-b"]
    assert history.frame["observed_round"].tolist() == [0, 0, 1]
    assert history.frame["objective_value"].tolist() == pytest.approx([1.0, 0.0, 0.5])
    assert history.summary.to_dict(orient="records") == [
        {
            "observed_round": 0,
            "batch_id": "batch-0",
            "candidate_count": 2,
            "batch_median": 0.5,
            "between_candidate_q25": 0.25,
            "between_candidate_q75": 0.75,
            "cumulative_best": 1.0,
        },
        {
            "observed_round": 1,
            "batch_id": "batch-1",
            "candidate_count": 1,
            "batch_median": 0.5,
            "between_candidate_q25": 0.5,
            "between_candidate_q75": 0.5,
            "cumulative_best": 1.0,
        },
    ]


@pytest.mark.parametrize(
    "second_run_kwargs",
    [
        {"calibration": {**_CALIBRATION, "response_separation_scale": 2.0}},
        {"target_mask": [0, 0, 1, 1]},
        {"y_space": "another_vector_space_v1"},
        {"score_channel": "response_separation"},
    ],
)
def test_history_rejects_noncommensurate_semantics(
    tmp_path: Path,
    second_run_kwargs: dict[str, object],
) -> None:
    outputs_dir = tmp_path / "outputs"
    event = _event(
        "candidate-a",
        observed_round=0,
        batch_id="batch-0",
        y_obs=[0.0, 2.0, 0.0, 2.0, -1.0, 1.0, -1.0, 1.0],
    )
    rows = [
        _run_row(outputs_dir, round_k=0, run_id="r0", events=[event]),
        _run_row(outputs_dir, round_k=1, run_id="r1", events=[event], **second_run_kwargs),
    ]
    ledger_dir = outputs_dir / "ledger"
    ledger_dir.mkdir(parents=True)
    pl.DataFrame(rows).write_parquet(ledger_dir / "runs.parquet")
    run_series = {
        "schema_version": "opal.observed_objective_run_series.v1",
        "runs": [
            {
                "as_of_round": round_k,
                "run_id": run_id,
                "contract_sha256": observed_objective_run_contract_sha256(
                    outputs_dir=outputs_dir,
                    selection_view_id="ethanol",
                    as_of_round=round_k,
                    run_id=run_id,
                ),
            }
            for round_k, run_id in [(0, "r0"), (1, "r1")]
        ],
    }

    with pytest.raises(OpalError, match="not commensurate"):
        load_observed_objective_history(
            outputs_dir=outputs_dir,
            selection_view_id="ethanol",
            run_series=run_series,
        )


def test_history_rejects_a_changed_event_in_a_later_snapshot(tmp_path: Path) -> None:
    outputs_dir = tmp_path / "outputs"
    original = _event(
        "candidate-a",
        observed_round=0,
        batch_id="batch-0",
        y_obs=[0.0, 2.0, 0.0, 2.0, -1.0, 1.0, -1.0, 1.0],
    )
    changed = {**original, "y_obs": [0.0, 1.0, 0.0, 1.0, -1.0, 1.0, -1.0, 1.0]}
    rows = [
        _run_row(outputs_dir, round_k=0, run_id="r0", events=[original]),
        _run_row(outputs_dir, round_k=1, run_id="r1", events=[changed]),
    ]
    ledger_dir = outputs_dir / "ledger"
    ledger_dir.mkdir(parents=True)
    pl.DataFrame(rows).write_parquet(ledger_dir / "runs.parquet")
    run_series = {
        "schema_version": "opal.observed_objective_run_series.v1",
        "runs": [
            {
                "as_of_round": round_k,
                "run_id": run_id,
                "contract_sha256": observed_objective_run_contract_sha256(
                    outputs_dir=outputs_dir,
                    selection_view_id="ethanol",
                    as_of_round=round_k,
                    run_id=run_id,
                ),
            }
            for round_k, run_id in [(0, "r0"), (1, "r1")]
        ],
    }

    with pytest.raises(OpalError, match="changed across cumulative"):
        load_observed_objective_history(
            outputs_dir=outputs_dir,
            selection_view_id="ethanol",
            run_series=run_series,
        )


def test_history_rejects_a_later_snapshot_that_drops_prior_events(tmp_path: Path) -> None:
    outputs_dir = tmp_path / "outputs"
    first = _event(
        "candidate-a",
        observed_round=0,
        batch_id="batch-0",
        y_obs=[0.0, 2.0, 0.0, 2.0, -1.0, 1.0, -1.0, 1.0],
    )
    second = _event(
        "candidate-b",
        observed_round=1,
        batch_id="batch-1",
        y_obs=[0.0, 1.0, 0.0, 1.0, -0.5, 0.5, -0.5, 0.5],
    )
    rows = [
        _run_row(outputs_dir, round_k=0, run_id="r0", events=[first]),
        _run_row(outputs_dir, round_k=1, run_id="r1", events=[second]),
    ]
    ledger_dir = outputs_dir / "ledger"
    ledger_dir.mkdir(parents=True)
    pl.DataFrame(rows).write_parquet(ledger_dir / "runs.parquet")
    run_series = {
        "schema_version": "opal.observed_objective_run_series.v1",
        "runs": [
            {
                "as_of_round": round_k,
                "run_id": run_id,
                "contract_sha256": observed_objective_run_contract_sha256(
                    outputs_dir=outputs_dir,
                    selection_view_id="ethanol",
                    as_of_round=round_k,
                    run_id=run_id,
                ),
            }
            for round_k, run_id in [(0, "r0"), (1, "r1")]
        ],
    }

    with pytest.raises(OpalError, match="drops prior events"):
        load_observed_objective_history(
            outputs_dir=outputs_dir,
            selection_view_id="ethanol",
            run_series=run_series,
        )


def test_history_rejects_a_run_contract_digest_mismatch(tmp_path: Path) -> None:
    outputs_dir = tmp_path / "outputs"
    event = _event(
        "candidate-a",
        observed_round=0,
        batch_id="batch-0",
        y_obs=[0.0, 2.0, 0.0, 2.0, -1.0, 1.0, -1.0, 1.0],
    )
    row = _run_row(outputs_dir, round_k=0, run_id="r0", events=[event])
    ledger_dir = outputs_dir / "ledger"
    ledger_dir.mkdir(parents=True)
    pl.DataFrame([row]).write_parquet(ledger_dir / "runs.parquet")

    with pytest.raises(OpalError, match="digest mismatch"):
        load_observed_objective_history(
            outputs_dir=outputs_dir,
            selection_view_id="ethanol",
            run_series={
                "schema_version": "opal.observed_objective_run_series.v1",
                "runs": [{"as_of_round": 0, "run_id": "r0", "contract_sha256": "0" * 64}],
            },
        )
