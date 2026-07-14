"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/storage/test_observed_label_store.py

Contracts for shared observed-label sources.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dnadesign.opal.src.core.utils import OpalError
from dnadesign.opal.src.storage.label_sources import ObservedLabelStore
from dnadesign.opal.src.storage.locks import PathLock


def _write_labels(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "id": ["a", "b", "a", "c"],
            "observed_round": [0, 0, 1, 2],
            "batch_id": ["batch0", "batch0", "batch1", "batch2"],
            "y_space": ["sfxi_vec8", "sfxi_vec8", "sfxi_vec8", "sfxi_vec8"],
            "y_obs": [[0.0], [1.0], [0.5], [9.0]],
            "src": ["assay", "assay", "assay", "assay"],
            "ts": [
                "2026-05-17T00:00:00Z",
                "2026-05-17T00:01:00Z",
                "2026-05-17T00:02:00Z",
                "2026-05-17T00:03:00Z",
            ],
        }
    ).to_parquet(path, index=False)


def test_observed_label_store_trains_latest_labels_through_round(tmp_path: Path) -> None:
    labels_path = tmp_path / "observed_labels.parquet"
    _write_labels(labels_path)
    store = ObservedLabelStore(
        path=labels_path,
        y_space="sfxi_vec8",
        id_column="id",
        round_column="observed_round",
        batch_column="batch_id",
        dedup_policy="latest_by_round",
    )

    out = store.training_labels(
        as_of_round=1,
        cumulative_training=True,
        dedup_policy="latest_only",
        known_ids={"a", "b", "c"},
    )

    assert out["id"].tolist() == ["a", "b"]
    assert out["r"].tolist() == [1, 0]
    assert out["y"].tolist() == [[0.5], [1.0]]


def test_observed_label_store_rejects_unknown_candidate_ids(tmp_path: Path) -> None:
    labels_path = tmp_path / "observed_labels.parquet"
    _write_labels(labels_path)
    store = ObservedLabelStore(
        path=labels_path,
        y_space="sfxi_vec8",
        id_column="id",
        round_column="observed_round",
        batch_column="batch_id",
        dedup_policy="latest_by_round",
    )

    with pytest.raises(OpalError, match="unknown ids"):
        store.training_labels(
            as_of_round=1,
            cumulative_training=True,
            dedup_policy="latest_only",
            known_ids={"a"},
        )


def test_observed_label_store_reports_observed_ids_by_round(tmp_path: Path) -> None:
    labels_path = tmp_path / "observed_labels.parquet"
    _write_labels(labels_path)
    store = ObservedLabelStore(
        path=labels_path,
        y_space="sfxi_vec8",
        id_column="id",
        round_column="observed_round",
        batch_column="batch_id",
        dedup_policy="latest_by_round",
    )

    assert store.observed_ids(as_of_round=0, known_ids={"a", "b", "c"}) == {"a", "b"}
    assert store.observed_ids(as_of_round=1, known_ids={"a", "b", "c"}) == {"a", "b"}


def test_observed_label_store_appends_and_replaces_round_labels(tmp_path: Path) -> None:
    labels_path = tmp_path / "observed_labels.parquet"
    store = ObservedLabelStore(
        path=labels_path,
        y_space="sfxi_vec8",
        id_column="id",
        round_column="observed_round",
        batch_column="batch_id",
        dedup_policy="latest_by_round",
    )

    written = store.append_labels(
        pd.DataFrame({"id": ["a"], "y": [[0.25]]}),
        observed_round=0,
        batch_id="batch0",
        src="test",
        if_exists="fail",
        known_ids={"a"},
    )
    assert written.to_dict(orient="records") == [{"id": "a", "y": [0.25]}]

    replaced = store.append_labels(
        pd.DataFrame({"id": ["a"], "y": [[0.75]]}),
        observed_round=0,
        batch_id="batch0b",
        src="test",
        if_exists="replace",
        known_ids={"a"},
    )
    assert replaced.to_dict(orient="records") == [{"id": "a", "y": [0.75]}]
    out = store.training_labels(
        as_of_round=0,
        cumulative_training=True,
        dedup_policy="latest_only",
        known_ids={"a"},
    )
    assert out["y"].tolist() == [[0.75]]


def test_observed_label_store_append_uses_sidecar_path_lock(tmp_path: Path) -> None:
    labels_path = tmp_path / "observed_labels.parquet"
    store = ObservedLabelStore(
        path=labels_path,
        y_space="sfxi_vec8",
        id_column="id",
        round_column="observed_round",
        batch_column="batch_id",
        dedup_policy="latest_by_round",
    )

    with PathLock(labels_path, lock_name="Observed label source"):
        with pytest.raises(OpalError, match="Observed label source is locked"):
            store.append_labels(
                pd.DataFrame({"id": ["a"], "y": [[0.25]]}),
                observed_round=0,
                batch_id="batch0",
                src="test",
                if_exists="fail",
                known_ids={"a"},
            )

    written = store.append_labels(
        pd.DataFrame({"id": ["a"], "y": [[0.25]]}),
        observed_round=0,
        batch_id="batch0",
        src="test",
        if_exists="fail",
        known_ids={"a"},
    )

    assert written.to_dict(orient="records") == [{"id": "a", "y": [0.25]}]
    assert not labels_path.with_name(".observed_labels.parquet.lock").exists()
