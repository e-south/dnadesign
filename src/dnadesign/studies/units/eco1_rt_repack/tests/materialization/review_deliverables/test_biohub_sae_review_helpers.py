"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/test_biohub_sae_review_helpers.py

Biohub ESMC SAE review-helper tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables import (
    biohub_esmc_sae_interpretation_shared as sae_shared,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables import (
    biohub_esmc_sae_tables,
    sae_structure_browser,
)


def test_top_sae_feature_prevalence_uses_activation_threshold(tmp_path: Path) -> None:
    protein_features_path = tmp_path / "protein_features.parquet"
    residue_features_path = tmp_path / "residue_features.parquet"
    feature_catalog_path = tmp_path / "feature_catalog.parquet"
    table_path = tmp_path / "top_features.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                _protein_feature_row(1, nonzero=4, total=0.02, max_value=0.005),
                _protein_feature_row(2, nonzero=2, total=0.40, max_value=0.20),
                _protein_feature_row(3, nonzero=1, total=1.00, max_value=1.00),
            ]
        ),
        protein_features_path,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                *[_residue_feature_row(1, position, 0.005) for position in range(1, 5)],
                _residue_feature_row(2, 1, 0.20),
                _residue_feature_row(2, 2, 0.20),
                _residue_feature_row(3, 6, 1.00),
            ]
        ),
        residue_features_path,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "sae_model": "fixture",
                    "feature_index": feature_index,
                    "label": "",
                    "description": "",
                }
                for feature_index in (1, 2, 3)
            ]
        ),
        feature_catalog_path,
    )

    biohub_esmc_sae_tables.write_protein_top_feature_table(
        path=table_path,
        protein_features_path=protein_features_path,
        residue_features_path=residue_features_path,
        feature_catalog_path=feature_catalog_path,
        top_n=1,
    )

    rows = {row["feature_index"]: row for row in pq.read_table(table_path).to_pylist()}
    assert 1 not in rows
    assert rows[2]["rank_by_prevalence"] == 1
    assert rows[2]["prevalent_residue_count"] == 2
    assert rows[2]["prevalence_activation_threshold"] == biohub_esmc_sae_tables.FEATURE_PREVALENCE_THRESHOLD
    assert rows[3]["rank_by_max_activation"] == 1


def test_sae_feature_labels_stay_single_line() -> None:
    label = sae_shared.feature_axis_label(
        101,
        "Polymerase thumb region",
        "Fixture exact-dictionary feature description for a polymerase-like region.",
    )
    assert label == "F101 - Polymerase thumb region"
    assert "\n" not in label
    assert len(label) <= 66


def test_sae_structure_browser_descriptions_stay_concise() -> None:
    description = (
        "Summary: Right-hand nucleic-acid polymerase module, with a strong preference for the C-terminal "
        "helical thumb/CTE and adjacent palm region that contacts and positions the template-product duplex. "
        "Activation pattern: many long source details that belong in the SAE feature inspector, not the "
        "structure-browser title region."
    )

    concise = sae_structure_browser._concise_sae_description(description)

    assert concise.startswith("Right-hand nucleic-acid polymerase module")
    assert "Activation pattern" not in concise
    assert len(concise) <= 261


def _protein_feature_row(
    feature_index: int,
    *,
    nonzero: int,
    total: float,
    max_value: float,
) -> dict[str, object]:
    return {
        "candidate_id": "wild_type",
        "sequence_hash": "sha256:" + "1" * 64,
        "sae_model": "fixture",
        "feature_index": feature_index,
        "sequence_residue_count": 6,
        "nonzero_residue_count": nonzero,
        "activation_sum": total,
        "activation_mean": total / 6.0,
        "activation_max": max_value,
    }


def _residue_feature_row(feature_index: int, position: int, value: float) -> dict[str, object]:
    return {
        "candidate_id": "wild_type",
        "sequence_hash": "sha256:" + "1" * 64,
        "sae_model": "fixture",
        "residue_index_zero_based": position - 1,
        "sequence_position_one_based": position,
        "feature_index": feature_index,
        "value": value,
    }
