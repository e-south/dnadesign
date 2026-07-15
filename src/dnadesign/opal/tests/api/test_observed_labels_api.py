"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/api/test_observed_labels_api.py

Contracts for OPAL's public immutable observed-label snapshot API.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from dnadesign.opal import (
    OBSERVED_LABEL_PROMOTION_SCHEMA_VERSION,
    ObservedLabelPromotionBinding,
    verify_observed_label_snapshot,
)

LABEL_RELATIVE_PATH = "_opal/observed_labels.parquet"
MANIFEST_RELATIVE_PATH = "_opal/observed_labels.manifest.json"
PROVENANCE_RELATIVE_PATH = "_opal/study_label_provenance.json"
CANDIDATE_RELATIVE_PATH = "records.parquet"


def test_public_snapshot_api_verifies_and_materializes_exact_vectors(tmp_path: Path) -> None:
    _write_promotion(tmp_path, _label_frame())

    snapshot = verify_observed_label_snapshot(_binding(tmp_path), expected_y_width=8)

    assert snapshot.promotion.row_count == 2
    assert snapshot.labels.to_dict(orient="records") == [
        {"id": "candidate_a", "y": [0.0] * 8, "r": 0},
        {"id": "candidate_b", "y": [1.0] * 8, "r": 0},
    ]


def test_public_snapshot_api_rejects_vector_width_mismatch(tmp_path: Path) -> None:
    _write_promotion(tmp_path, _label_frame())

    with pytest.raises(ValueError, match="expected 4 values"):
        verify_observed_label_snapshot(_binding(tmp_path), expected_y_width=4)


def test_public_snapshot_api_materializes_the_complete_round_domain(tmp_path: Path) -> None:
    frame = _label_frame()
    frame.loc[0, "observed_round"] = 2**40
    _write_promotion(tmp_path, frame)

    snapshot = verify_observed_label_snapshot(_binding(tmp_path), expected_y_width=8)

    assert snapshot.labels.set_index("id")["r"].to_dict() == {
        "candidate_a": 2**40,
        "candidate_b": 0,
    }


@pytest.mark.parametrize("duplicate_round", [0, 1], ids=["same-round", "cross-round"])
def test_public_snapshot_api_rejects_duplicate_candidate_labels(tmp_path: Path, duplicate_round: int) -> None:
    duplicate = (
        _label_frame()
        .iloc[[0]]
        .assign(
            observed_round=duplicate_round,
            batch_id=f"batch_{duplicate_round}",
        )
    )
    frame = pd.concat([_label_frame(), duplicate], ignore_index=True)
    _write_promotion(tmp_path, frame)

    with pytest.raises(ValueError, match="[Dd]uplicate"):
        verify_observed_label_snapshot(_binding(tmp_path), expected_y_width=8)


@pytest.mark.parametrize("expected_y_width", [0, -1, True, 8.0])
def test_public_snapshot_api_rejects_invalid_expected_width(
    tmp_path: Path,
    expected_y_width: object,
) -> None:
    _write_promotion(tmp_path, _label_frame())

    with pytest.raises(ValueError, match="positive integer"):
        verify_observed_label_snapshot(_binding(tmp_path), expected_y_width=expected_y_width)  # type: ignore[arg-type]


def _label_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ["candidate_a", "candidate_b"],
            "observed_round": [0, 0],
            "batch_id": ["batch_0", "batch_0"],
            "y_space": ["response_window_vector_v1", "response_window_vector_v1"],
            "y_obs": [[0.0] * 8, [1.0] * 8],
        }
    )


def _write_promotion(dataset_root: Path, labels: pd.DataFrame) -> None:
    candidate_path = dataset_root / CANDIDATE_RELATIVE_PATH
    pd.DataFrame(
        {
            "id": ["candidate_a", "candidate_b"],
            "sequence": ["ACGT", "TGCA"],
            "x_feature": [[0.0, 1.0], [1.0, 0.0]],
        }
    ).to_parquet(candidate_path, index=False)
    label_path = dataset_root / LABEL_RELATIVE_PATH
    label_path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(label_path, index=False)
    provenance_path = dataset_root / PROVENANCE_RELATIVE_PATH
    provenance_path.write_text('{"schema_version":"stress-study.labels.v1"}\n', encoding="utf-8")
    payload = {
        "schema_version": OBSERVED_LABEL_PROMOTION_SCHEMA_VERSION,
        "campaign_slug": "promoter",
        "study_id": "stress_promoter",
        "y_space": "response_window_vector_v1",
        "study_provenance": {
            "schema_id": "stress-study.labels.v1",
            "path": PROVENANCE_RELATIVE_PATH,
            "sha256": _sha256(provenance_path),
        },
        "candidate_artifact": _candidate_artifact(candidate_path),
        "label_artifact": {
            "path": LABEL_RELATIVE_PATH,
            "sha256": _sha256(label_path),
            "row_count": len(labels),
        },
    }
    (dataset_root / MANIFEST_RELATIVE_PATH).write_text(json.dumps(payload), encoding="utf-8")


def _binding(dataset_root: Path) -> ObservedLabelPromotionBinding:
    return ObservedLabelPromotionBinding(
        dataset_root=dataset_root,
        manifest_path=MANIFEST_RELATIVE_PATH,
        label_path=LABEL_RELATIVE_PATH,
        campaign_slug="promoter",
        study_id="stress_promoter",
        y_space="response_window_vector_v1",
        candidate_x_column="x_feature",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate_artifact(path: Path) -> dict[str, object]:
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    return {
        "path": CANDIDATE_RELATIVE_PATH,
        "sha256": _sha256(path),
        "row_count": int(parquet.metadata.num_rows),
        "columns": schema.names,
        "schema_sha256": hashlib.sha256(schema.serialize().to_pybytes()).hexdigest(),
    }
