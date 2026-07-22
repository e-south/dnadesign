"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/sae_window_summary/test_materialization.py

SAE window summary materialization tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.sae_window_summary import (
    WindowSpec,
    materialize_sae_window_summary,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.sae_window_summary.windows import (
    default_window_specs,
)


def test_sae_window_summary_compares_local_window_vectors_to_wt(tmp_path: Path) -> None:
    residue_features_path = tmp_path / "biohub_esmc_residue_features.parquet"
    profile_path = tmp_path / "biohub_esmc_sae_profile.parquet"
    feature_catalog_path = tmp_path / "biohub_esmc_feature_catalog.parquet"
    candidate_pool_path = tmp_path / "candidate_pool.parquet"
    output_root = tmp_path / "design_classes"
    _write_table(
        residue_features_path,
        [
            _feature("wild_type", "sha256:wt", 1, 10, 1.0),
            _feature("wild_type", "sha256:wt", 2, 10, 1.0),
            _feature("wild_type", "sha256:wt", 3, 20, 0.5),
            _feature("candidate_a", "sha256:a", 1, 10, 2.0),
            _feature("candidate_a", "sha256:a", 2, 10, 2.0),
            _feature("candidate_a", "sha256:a", 3, 20, 0.5),
            _feature("candidate_b", "sha256:b", 1, 10, 1.0),
            _feature("candidate_b", "sha256:b", 2, 10, 1.0),
            _feature("candidate_b", "sha256:b", 3, 20, 1.5),
        ],
    )
    _write_table(
        profile_path,
        [
            _profile("wild_type", "sha256:wt"),
            _profile("candidate_a", "sha256:a"),
            _profile("candidate_b", "sha256:b"),
        ],
    )
    _write_table(
        feature_catalog_path,
        [
            {"sae_model": "toy_sae", "feature_index": 10, "label": "Feature 10", "description": "short label"},
            {"sae_model": "toy_sae", "feature_index": 20, "label": "Feature 20", "description": "short label"},
        ],
    )
    _write_table(
        candidate_pool_path,
        [
            {"candidate_id": "candidate_a", "design_class_id": "class_a"},
            {"candidate_id": "candidate_b", "design_class_id": "class_b"},
        ],
    )

    result = materialize_sae_window_summary(
        repo_root=tmp_path,
        output_root=output_root,
        residue_features_path=residue_features_path,
        profile_path=profile_path,
        feature_catalog_path=feature_catalog_path,
        candidate_pool_path=candidate_pool_path,
        window_specs=(
            WindowSpec("toy_surface", "Toy surface", (1, 2), "test window"),
            WindowSpec("toy_control", "Toy control", (3,), "test window"),
        ),
    )

    rows = pq.read_table(result.summary_path).to_pylist()
    by_key = {(str(row["candidate_id"]), str(row["window_id"])): row for row in rows}
    assert by_key[("wild_type", "toy_surface")]["cosine_distance_to_wt"] == 0.0
    assert by_key[("candidate_a", "toy_surface")]["activation_delta_sum_vs_wt"] == 2.0
    assert by_key[("candidate_a", "toy_surface")]["window_redundancy_group"] in {
        "near_duplicate_window",
        "close_window",
        "distinct_window",
    }
    top_deltas = json.loads(by_key[("candidate_a", "toy_surface")]["top5_signed_feature_deltas_json"])
    assert top_deltas[0]["feature_index"] == 10
    assert top_deltas[0]["activation_delta_vs_wt"] == 2.0
    assert by_key[("candidate_b", "toy_control")]["design_class_id"] == "class_b"
    assert result.manifest_path.exists()


def test_default_window_specs_include_basic_surface_positions() -> None:
    specs = default_window_specs(
        [
            _mask_row(190, "D", False, 4.0),
            _mask_row(20, "S", True, 8.0),
            _mask_row(30, "K", True, 18.0),
            _mask_row(40, "R", False, 18.0),
            _mask_row(50, "H", True, None),
        ]
    )

    by_id = {spec.window_id: spec for spec in specs}
    annulus = by_id["mutable_substrate_proximal_annulus_basic_surface"]
    assert annulus.residue_positions_1based == (20, 30)


def _feature(
    candidate_id: str, sequence_hash: str, position: int, feature_index: int, value: float
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "sequence_hash": sequence_hash,
        "sae_model": "toy_sae",
        "residue_index_zero_based": position - 1,
        "sequence_position_one_based": position,
        "feature_index": feature_index,
        "value": value,
    }


def _profile(candidate_id: str, sequence_hash: str) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "sequence_hash": sequence_hash,
        "model": "toy_model",
        "sae_model": "toy_sae",
        "sequence_length": 3,
        "feature_dictionary_size": 32,
        "status": "accepted",
    }


def _mask_row(position: int, wt_aa: str, non_fixed: bool, distance: float | None) -> dict[str, object]:
    return {
        "canonical_position": position,
        "wt_aa": wt_aa,
        "mapping_status": "mapped",
        "non_fixed": non_fixed,
        "min_distance_to_retained_dna_rna_angstrom": distance,
    }


def _write_table(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)
