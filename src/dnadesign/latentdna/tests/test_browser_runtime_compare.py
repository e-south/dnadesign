"""
Contract tests for the latentdna browser compare runtime helpers.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from dnadesign.latentdna.src.notebooks.browser_runtime_compare import compare_pair_payload, display_compare_basis


def _write_view(
    output_root: Path,
    *,
    view_id: str,
    rows: pd.DataFrame,
    matrix: np.ndarray | None = None,
    status: str = "ok",
    stale: bool = False,
) -> None:
    view_dir = output_root / "views" / view_id
    view_dir.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(view_dir / "rows.parquet", index=False)
    if matrix is not None:
        np.save(view_dir / "matrix.npy", matrix.astype(np.float32))
    view_dir.joinpath("manifest.json").write_text(
        json.dumps(
            {
                "artifact_kind": "view",
                "artifact_id": view_id,
                "status": status,
                "stale": stale,
            }
        ),
        encoding="utf-8",
    )


def _write_alignment(
    output_root: Path,
    *,
    alignment_id: str,
    rows: list[dict[str, object]],
    status: str = "ok",
    stale: bool = False,
) -> None:
    alignment_dir = output_root / "alignments" / alignment_id
    alignment_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(alignment_dir / "mapping.parquet", index=False)
    alignment_dir.joinpath("manifest.json").write_text(
        json.dumps(
            {
                "artifact_kind": "alignment_set",
                "artifact_id": alignment_id,
                "status": status,
                "stale": stale,
            }
        ),
        encoding="utf-8",
    )


def test_compare_pair_payload_falls_back_to_shared_key_basis(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    _write_view(
        output_root,
        view_id="left_view",
        rows=pd.DataFrame(
            {
                "construct__anchor_id": ["a", "b", "c"],
                "usr_label__primary": ["spyP", "sulAp", "soxSp"],
            }
        ),
        matrix=np.asarray(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ]
        ),
    )
    _write_view(
        output_root,
        view_id="right_view",
        rows=pd.DataFrame(
            {
                "construct__anchor_id": ["a", "b", "c"],
                "usr_label__primary": ["spyP", "sulAp", "soxSp"],
            }
        ),
        matrix=np.asarray(
            [
                [0.0, 1.0],
                [0.8, 0.2],
                [1.0, 0.9],
            ]
        ),
    )

    payload = compare_pair_payload(
        "left_view",
        "right_view",
        geometry_rows_by_id={
            "left_view": {"coordinate_space_id": "shared_space"},
            "right_view": {"coordinate_space_id": "shared_space"},
        },
        comparison_bases=[],
        compare_metrics={"sample_rows": 3, "distance_pair_limit": 8, "knn_k": 1},
        output_root=output_root,
    )

    assert payload["status"] == "ok"
    assert payload["basis"] == "shared_key:construct__anchor_id"
    assert payload["rows"] == 3
    assert payload["same_dims"] is True
    assert payload["same_coordinate_space"] is True
    assert payload["metrics"]["distance_spearman"] is not None
    assert payload["metrics"]["linear_cka"] is not None
    assert payload["metrics"]["neighbor_set_jaccard"] is not None
    assert payload["rowwise_cosine"] is not None
    assert payload["rowwise_diff_norm"] is not None


def test_compare_pair_payload_falls_back_from_duplicate_construct_anchor_id_to_id_join(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "outputs"
    _write_view(
        output_root,
        view_id="left_view",
        rows=pd.DataFrame(
            {
                "construct__anchor_id": ["dup", "dup", "c"],
                "id": ["a", "b", "c"],
                "usr_label__primary": ["spyP", "sulAp", "soxSp"],
            }
        ),
        matrix=np.asarray(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ]
        ),
    )
    _write_view(
        output_root,
        view_id="right_view",
        rows=pd.DataFrame(
            {
                "construct__anchor_id": ["dup", "dup", "c"],
                "id": ["a", "b", "c"],
                "usr_label__primary": ["spyP", "sulAp", "soxSp"],
            }
        ),
        matrix=np.asarray(
            [
                [0.0, 1.0],
                [0.8, 0.2],
                [1.0, 0.9],
            ]
        ),
    )

    payload = compare_pair_payload(
        "left_view",
        "right_view",
        geometry_rows_by_id={
            "left_view": {"coordinate_space_id": "shared_space"},
            "right_view": {"coordinate_space_id": "shared_space"},
        },
        comparison_bases=[],
        compare_metrics={"sample_rows": 3, "distance_pair_limit": 8, "knn_k": 1},
        output_root=output_root,
    )

    assert payload["status"] == "ok"
    assert payload["basis"] == "shared_key:id"
    assert payload["rows"] == 3


def test_compare_pair_payload_errors_when_no_unique_shared_key_exists(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    _write_view(
        output_root,
        view_id="left_view",
        rows=pd.DataFrame(
            {
                "construct__anchor_id": ["a", "a", "b"],
                "usr_label__primary": ["spyP", "sulAp", "soxSp"],
            }
        ),
    )
    _write_view(
        output_root,
        view_id="right_view",
        rows=pd.DataFrame(
            {
                "construct__anchor_id": ["a", "b", "c"],
                "usr_label__primary": ["spyP", "sulAp", "soxSp"],
            }
        ),
    )

    payload = compare_pair_payload(
        "left_view",
        "right_view",
        geometry_rows_by_id={},
        comparison_bases=[],
        compare_metrics={},
        output_root=output_root,
    )

    assert payload["status"] == "error"
    assert "shared unique row key" in str(payload["error"])


def test_compare_pair_payload_errors_when_preferred_unique_join_key_has_zero_overlap(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    _write_view(
        output_root,
        view_id="left_view",
        rows=pd.DataFrame(
            {
                "construct__anchor_id": ["anchor_a", "anchor_b", "anchor_c"],
                "id": ["shared_0", "shared_1", "shared_2"],
            }
        ),
        matrix=np.asarray(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ]
        ),
    )
    _write_view(
        output_root,
        view_id="right_view",
        rows=pd.DataFrame(
            {
                "construct__anchor_id": ["context_a", "context_b", "context_c"],
                "id": ["shared_0", "shared_1", "shared_2"],
            }
        ),
        matrix=np.asarray(
            [
                [0.0, 1.0],
                [0.8, 0.2],
                [1.0, 0.9],
            ]
        ),
    )

    payload = compare_pair_payload(
        "left_view",
        "right_view",
        geometry_rows_by_id={},
        comparison_bases=[],
        compare_metrics={"sample_rows": 3},
        output_root=output_root,
    )

    assert payload["status"] == "error"
    assert "refusing fallback to a weaker join basis" in str(payload["error"])
    assert "construct__anchor_id" in str(payload["error"])


def test_compare_pair_payload_refuses_stale_view_manifests(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    _write_view(
        output_root,
        view_id="left_view",
        rows=pd.DataFrame(
            {
                "construct__anchor_id": ["a", "b", "c"],
                "id": ["a", "b", "c"],
                "usr_label__primary": ["spyP", "sulAp", "soxSp"],
            }
        ),
        matrix=np.asarray(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ]
        ),
        stale=True,
    )
    _write_view(
        output_root,
        view_id="right_view",
        rows=pd.DataFrame(
            {
                "construct__anchor_id": ["a", "b", "c"],
                "id": ["a", "b", "c"],
                "usr_label__primary": ["spyP", "sulAp", "soxSp"],
            }
        ),
        matrix=np.asarray(
            [
                [0.0, 1.0],
                [0.8, 0.2],
                [1.0, 0.9],
            ]
        ),
    )

    payload = compare_pair_payload(
        "left_view",
        "right_view",
        geometry_rows_by_id={},
        comparison_bases=[],
        compare_metrics={},
        output_root=output_root,
    )

    assert payload["status"] == "missing"
    assert "view artifact is not fresh" in str(payload["error"])


def test_compare_pair_payload_refuses_non_fresh_explicit_alignment_manifest(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    _write_view(
        output_root,
        view_id="left_view",
        rows=pd.DataFrame({"id": ["row0", "row1"]}),
        matrix=np.asarray([[0.0, 1.0], [1.0, 0.0]]),
    )
    _write_view(
        output_root,
        view_id="right_view",
        rows=pd.DataFrame({"id": ["row0", "row1"]}),
        matrix=np.asarray([[0.0, 1.0], [0.8, 0.2]]),
    )
    _write_alignment(
        output_root,
        alignment_id="left_to_right",
        rows=[
            {"left_indices": [0], "right_indices": [0], "left_count": 1, "right_count": 1},
            {"left_indices": [1], "right_indices": [1], "left_count": 1, "right_count": 1},
        ],
        status="attention",
    )

    payload = compare_pair_payload(
        "left_view",
        "right_view",
        geometry_rows_by_id={},
        comparison_bases=[
            {
                "left_view": "left_view",
                "right_view": "right_view",
                "alignment_id": "left_to_right",
            }
        ],
        compare_metrics={"sample_rows": 2},
        output_root=output_root,
    )

    assert payload["status"] == "error"
    assert "alignment_set artifact is not fresh" in str(payload["error"])


def test_compare_pair_payload_errors_when_matrix_row_count_drift_exists(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    _write_view(
        output_root,
        view_id="left_view",
        rows=pd.DataFrame(
            {
                "construct__anchor_id": ["a", "b", "c"],
                "id": ["a", "b", "c"],
            }
        ),
        matrix=np.asarray(
            [
                [0.0, 1.0],
                [1.0, 0.0],
            ]
        ),
    )
    _write_view(
        output_root,
        view_id="right_view",
        rows=pd.DataFrame(
            {
                "construct__anchor_id": ["a", "b", "c"],
                "id": ["a", "b", "c"],
            }
        ),
        matrix=np.asarray(
            [
                [0.0, 1.0],
                [0.8, 0.2],
                [1.0, 0.9],
            ]
        ),
    )

    payload = compare_pair_payload(
        "left_view",
        "right_view",
        geometry_rows_by_id={},
        comparison_bases=[],
        compare_metrics={"sample_rows": 3},
        output_root=output_root,
    )

    assert payload["status"] == "error"
    assert "row count does not match rows artifact" in str(payload["error"])


def test_display_compare_basis_preserves_asymmetric_shared_key_labels() -> None:
    label = display_compare_basis("shared_key:id->construct__anchor_id")

    assert label == "Shared key: ID -> Construct Anchor ID"
