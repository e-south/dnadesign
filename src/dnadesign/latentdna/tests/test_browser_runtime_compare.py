"""
Contract tests for the latentdna browser compare runtime helpers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dnadesign.latentdna.src.notebooks.browser_runtime_compare import compare_pair_payload


def _write_view(output_root: Path, *, view_id: str, rows: pd.DataFrame, matrix: np.ndarray | None = None) -> None:
    view_dir = output_root / "views" / view_id
    view_dir.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(view_dir / "rows.parquet", index=False)
    if matrix is not None:
        np.save(view_dir / "matrix.npy", matrix.astype(np.float32))


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
    assert payload["metrics"]["mean_knn_overlap"] is not None
    assert payload["rowwise_cosine"] is not None
    assert payload["rowwise_diff_norm"] is not None


def test_compare_pair_payload_rejects_duplicate_shared_key_rows(tmp_path: Path) -> None:
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
    assert "shared-key comparison requires unique" in str(payload["error"])
