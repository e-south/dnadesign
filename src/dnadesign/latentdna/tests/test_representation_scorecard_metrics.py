"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/tests/test_representation_scorecard_metrics.py

Regression tests for representation scorecard metrics LatentDNA.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from types import SimpleNamespace

import pyarrow as pa
import pytest

from dnadesign.latentdna.src.contracts.errors import ContractViolationError
from dnadesign.latentdna.src.io.parquet_io import write_table
from dnadesign.latentdna.src.scalars.builders.representation_scorecard import _representation_scorecard_table


def test_scorecard_uses_explicit_tasks_and_reads_each_score_table_once(tmp_path, monkeypatch) -> None:
    from dnadesign.latentdna.src.scalars.builders import representation_scorecard as scorecard

    path = tmp_path / "scalars/scores/table.parquet"
    path.parent.mkdir(parents=True)
    write_table(
        pa.table(
            {"group": ["a", "b", "both", "none"], "left": [0.9, -0.2, 0.8, -0.4], "right": [-0.3, 0.9, 0.8, -0.4]}
        ),
        path,
    )
    definition = {
        "mathematical_definition": "AUROC using explicit score and label columns.",
        "metric_family": "classification",
        "evidence_tier": "primary",
        "unit": "auroc",
        "direction": "higher_is_better",
        "aggregation_level": "candidate_summary",
    }
    config = SimpleNamespace(
        metric_definitions={key: {**definition, "display_name": key} for key in ("a_auc", "joint_auc")}
    )
    read = scorecard.read_table
    calls = []

    def counted_read(source):
        calls.append(source)
        return read(source)

    monkeypatch.setattr(scorecard, "read_table", counted_read)
    table, inputs, _ = scorecard._representation_scorecard_table(
        SimpleNamespace(output_root=tmp_path, config=config),
        candidates=[
            {
                "candidate_id": "encoder",
                "classification_tasks": [
                    {
                        "source_scalar": "scores",
                        "label_column": "group",
                        "positive_values": ["a", "both"],
                        "score_columns": ["left"],
                        "metrics": {"auroc": "a_auc"},
                    },
                    {
                        "source_scalar": "scores",
                        "label_column": "group",
                        "positive_values": ["both"],
                        "score_columns": ["left", "right"],
                        "metrics": {"auroc": "joint_auc"},
                    },
                ],
            }
        ],
    )
    assert {row["metric_id"]: row["value"] for row in table.to_pylist()} == {"a_auc": 1.0, "joint_auc": 1.0}
    assert calls == [path]
    assert len(inputs) == 1


def test_scorecard_rejects_implicit_study_inputs(tmp_path) -> None:
    with pytest.raises(ContractViolationError, match="classification_tasks"):
        _representation_scorecard_table(
            SimpleNamespace(output_root=tmp_path), candidates=[{"candidate_id": "encoder", "wildtype_source": "scores"}]
        )


def test_scorecard_rejects_ambiguous_metric_ids_and_obsolete_parameters(tmp_path):
    from dnadesign.latentdna.src.scalars.builders.representation_scorecard import (
        _classification_task_metrics,
        build_representation_scorecard_scalar,
    )

    with pytest.raises(ContractViolationError, match="metric ids"):
        _classification_task_metrics(
            [{"label": "a", "score": 0.5}, {"label": "b", "score": 0.1}],
            {
                "label_column": "label",
                "positive_values": ["a"],
                "score_columns": ["score"],
                "metrics": {"auroc": "same_id", "auprc": "same_id"},
            },
        )
    with pytest.raises(ContractViolationError, match="classification_tasks"):
        build_representation_scorecard_scalar(
            SimpleNamespace(), artifact_dir=tmp_path, params={"candidates": [], "label_column": "group"}
        )


@pytest.mark.parametrize(
    "params",
    [
        {"candidates": [{"candidate_id": "encoder", "classification_task": []}]},
        {"candidates": []},
        {"candidates": [{"candidate_id": "encoder"}]},
        {"candidates": [{"candidate_id": "encoder", "reducer_id": "missing"}], "refernece_neighbor_tasks": []},
    ],
)
def test_scorecard_rejects_invalid_configuration_before_writing(tmp_path, params):
    from dnadesign.latentdna.src.scalars.builders.representation_scorecard import build_representation_scorecard_scalar

    with pytest.raises(ContractViolationError, match="configuration"):
        build_representation_scorecard_scalar(
            SimpleNamespace(output_root=tmp_path), artifact_dir=tmp_path, params=params
        )
    assert not (tmp_path / "table.parquet").exists()
