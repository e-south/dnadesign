"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/conservation_fixtures.py

Conservation input fixtures for Eco1 review-deliverable tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


def write_conservation_inputs(output_root: Path) -> None:
    alignment_root = output_root / "conservation_alignments"
    alignment_root.mkdir(parents=True, exist_ok=True)
    source_root = output_root / "conservation_sources"
    source_root.mkdir(parents=True, exist_ok=True)
    alignment_root.joinpath("ec86_clade9_conservation_v1.aligned.fasta").write_text(
        "\n".join(
            [
                ">eco1_rt_ec86kit_reference",
                "MKSAYL",
                ">clade9_neighbor_001",
                "MKSAYL",
                ">clade9_neighbor_002",
                "MKSAFL",
                ">clade9_neighbor_003",
                "MRSAYI",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    alignment_root.joinpath("ec86_iia3_cluster42_1_conservation_v1.aligned.fasta").write_text(
        "\n".join(
            [
                ">eco1_rt_ec86kit_reference",
                "MKSAYL",
                ">ec86_iia3_cluster42_1_conservation_v1__001__001",
                "MKSAYL",
                ">ec86_iia3_cluster42_1_conservation_v1__002__002",
                "MKSAFL",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_conservation_source_manifest(source_root)
    _write_conservation_profile(output_root / "conservation_profile.parquet")


def _write_conservation_source_manifest(source_root: Path) -> None:
    source_root.joinpath("ec86_clade9_conservation_v1.source_manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_id": "eco1_rt_repack.conservation_source_sequence_bundle.profile",
                "schema_version": 1,
                "status": "materialized",
                "profile_id": "ec86_clade9_conservation_v1",
                "included_record_count": 3,
                "target_row_id": "eco1_rt_ec86kit_reference",
                "included_records": [
                    {
                        "record_id": f"clade9_neighbor_{index:03d}",
                        "provider_id": "fixture_provider",
                        "accession": f"fig|fixture.{index}.peg.1",
                    }
                    for index in range(1, 4)
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    source_root.joinpath("ec86_iia3_cluster42_1_conservation_v1.source_manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_id": "eco1_rt_repack.conservation_source_sequence_bundle.profile",
                "schema_version": 1,
                "status": "materialized",
                "profile_id": "ec86_iia3_cluster42_1_conservation_v1",
                "included_record_count": 2,
                "target_row_id": "eco1_rt_ec86kit_reference",
                "included_records": [
                    {
                        "record_id": f"ec86_iia3_cluster42_1_conservation_v1__{index:03d}__{index:03d}",
                        "provider_id": "fixture_provider",
                        "accession": f"fig|fixture.{index}.peg.1",
                    }
                    for index in range(1, 3)
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _write_conservation_profile(path: Path) -> None:
    rows = []
    for profile_id, non_gap_count, conserved_positions in (
        ("ec86_clade9_conservation_v1", 4, {2, 4}),
        ("ec86_iia3_cluster42_1_conservation_v1", 3, {1, 3, 5}),
    ):
        for position, wt_aa in enumerate("MKSAYL", start=1):
            rows.append(
                {
                    "canonical_position": position,
                    "profile_id": profile_id,
                    "wt_aa": wt_aa,
                    "msa_column": position,
                    "non_gap_count": non_gap_count,
                    "wt_count": max(1, non_gap_count - 1),
                    "wt_frequency": (non_gap_count - 1) / non_gap_count,
                    "plurality_aa": wt_aa,
                    "wt_is_plurality": True,
                    "conservation_threshold": 0.25,
                    "min_non_gap_count": 2,
                    "passes_conservation_mask": position in conserved_positions,
                    "source_hash": "sha256:" + "2" * 64,
                    "target_sequence_hash": "sha256:" + "3" * 64,
                    "mapping_status": "mapped",
                    "evidence_status": "used_for_mask" if position in conserved_positions else "not_conserved",
                }
            )
    pq.write_table(pa.Table.from_pylist(rows), path)
