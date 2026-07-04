"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/selection_readiness/_source_fixtures.py

Source-artifact fixtures for Eco1 panel-selection tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def write_selection_source_inputs(source_root: Path) -> None:
    """Write compact source artifacts needed by review-axis materialization."""

    source_root.mkdir(parents=True, exist_ok=True)
    _write_parquet(source_root / "conservation_profile.parquet", _conservation_rows())
    _write_parquet(source_root / "contact_geometry_profile.parquet", _contact_geometry_rows())
    (source_root / "mask_set.yaml").write_text(_mask_set_yaml(), encoding="utf-8")
    alignment_root = source_root / "conservation_alignments"
    alignment_root.mkdir(parents=True)
    _write_alignment(alignment_root / "ec86_clade9_conservation_v1.aligned.fasta", row_prefix="clade")
    _write_alignment(alignment_root / "ec86_iia3_cluster42_1_conservation_v1.aligned.fasta", row_prefix="subtype")


def _conservation_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for profile_id in ("ec86_clade9_conservation_v1", "ec86_iia3_cluster42_1_conservation_v1"):
        rows.extend(_profile_rows(profile_id))
    return rows


def _profile_rows(profile_id: str) -> list[dict[str, object]]:
    return [
        {
            "canonical_position": position,
            "profile_id": profile_id,
            "wt_aa": "A",
            "msa_column": position,
            "non_gap_count": 2,
            "wt_count": 1,
            "wt_frequency": 0.5,
            "plurality_aa": "A",
            "wt_is_plurality": True,
            "conservation_threshold": 0.25,
            "min_non_gap_count": 1,
            "passes_conservation_mask": False,
            "source_hash": "sha256:source",
            "target_sequence_hash": "sha256:target",
            "mapping_status": "mapped",
            "evidence_status": "usable",
        }
        for position in range(1, 221)
    ]


def _contact_geometry_rows() -> list[dict[str, object]]:
    return [
        {
            "canonical_position": position,
            "nearest_context_atom_distance_angstrom": 8.0 if position < 30 else 15.0,
        }
        for position in range(1, 221)
    ]


def _mask_set_yaml() -> str:
    rows = ["mask_policy_id: test_mask", "residues:"]
    for position in range(1, 221):
        rows.extend(
            [
                f"  - canonical_position: {position}",
                "    wang_ec86_direct_contact_prior: false",
            ]
        )
    return "\n".join(rows) + "\n"


def _write_alignment(path: Path, *, row_prefix: str) -> None:
    observed = list("A" * 220)
    for position in range(3, 12):
        observed[position - 1] = "G"
    for position in range(21, 30):
        observed[position - 1] = "V"
    path.write_text(
        "\n".join(
            [
                ">eco1_rt_ec86kit_reference",
                "A" * 220,
                f">{row_prefix}_source_1",
                "".join(observed),
                f">{row_prefix}_source_2",
                "A" * 220,
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_parquet(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)
