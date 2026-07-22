"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/mask_set/_fixtures.py

Shared mask-set materialization test fixtures.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.conservation import (
    materialize_conservation_profile,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.contact import materialize_contact_profile
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.contact_geometry import (
    materialize_contact_geometry_profile,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.structure import (
    materialize_structure_authority,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.structure_preprocessing import (
    materialize_structure_preprocessing_manifest,
)
from dnadesign.studies.units.eco1_rt_repack.tests._helpers import repo_root


def materialize_upstream_artifacts(tmp_path: Path) -> None:
    """Materialize the upstream Phase 1 artifacts needed by mask-set tests."""

    materialize_structure_authority(repo_root=repo_root(), output_root=tmp_path)
    materialize_structure_preprocessing_manifest(repo_root=repo_root(), output_root=tmp_path)
    materialize_contact_profile(repo_root=repo_root(), output_root=tmp_path)
    materialize_contact_geometry_profile(repo_root=repo_root(), output_root=tmp_path)
    materialize_conservation_profile(
        repo_root=repo_root(),
        output_root=tmp_path,
        alignment_root=_write_alignment_inputs(tmp_path),
    )


def _write_alignment_inputs(tmp_path: Path) -> Path:
    residue_rows = pq.read_table(tmp_path / "residue_map.parquet").to_pylist()
    target = "".join(str(row["wt_aa"]) for row in residue_rows)
    root = tmp_path / "alignments"
    root.mkdir()
    for profile_id in ("ec86_clade9_conservation_v1", "ec86_iia3_cluster42_1_conservation_v1"):
        records = [("eco1_rt_ec86kit_reference", target)]
        for index in range(20):
            sequence = list(target)
            sequence[2] = "S" if index < 14 else "A"
            sequence[3] = "G" if target[3] != "G" else "A"
            records.append((f"{profile_id}_homolog_{index + 1:02d}", "".join(sequence)))
        _write_fasta(root / f"{profile_id}.aligned.fasta", records)
    return root


def _write_fasta(path: Path, records: list[tuple[str, str]]) -> None:
    path.write_text(
        "".join(f">{record_id}\n{sequence}\n" for record_id, sequence in records),
        encoding="utf-8",
    )
