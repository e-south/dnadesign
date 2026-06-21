"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/conservation/test_materialization.py

Conservation-profile materialization tests for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.suite import validate_checked_in_contracts
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.conservation import (
    materialize_conservation_profile,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.contact import materialize_contact_profile
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.structure import (
    materialize_structure_authority,
)
from dnadesign.studies.units.eco1_rt_repack.tests._helpers import repo_root


def test_conservation_materializer_writes_long_form_profile(tmp_path: Path) -> None:
    materialize_structure_authority(repo_root=repo_root(), output_root=tmp_path)
    alignment_root = _write_alignment_inputs(tmp_path)

    result = materialize_conservation_profile(
        repo_root=repo_root(),
        output_root=tmp_path,
        alignment_root=alignment_root,
    )

    table = pq.read_table(result.conservation_profile_path)
    assert table.num_rows == 640
    metadata = table.schema.metadata or {}
    assert metadata[b"schema_id"] == b"thread.conservation_profile"
    assert metadata[b"status"] == b"materialized"
    assert metadata[b"profile_ids"] == b'["broad_tao_homolog_rt", "eco1_like_retron_rt"]'

    rows = table.to_pylist()
    position_3 = next(
        row for row in rows if row["profile_id"] == "broad_tao_homolog_rt" and row["canonical_position"] == 3
    )
    assert position_3["wt_aa"] == "S"
    assert position_3["msa_column"] == 3
    assert position_3["non_gap_count"] == 21
    assert position_3["wt_count"] == 15
    assert position_3["wt_is_plurality"] is True
    assert position_3["passes_conservation_mask"] is True

    position_4 = next(
        row for row in rows if row["profile_id"] == "broad_tao_homolog_rt" and row["canonical_position"] == 4
    )
    assert position_4["wt_is_plurality"] is False
    assert position_4["passes_conservation_mask"] is False

    position_1 = next(
        row for row in rows if row["profile_id"] == "broad_tao_homolog_rt" and row["canonical_position"] == 1
    )
    assert position_1["mapping_status"] == "unresolved_structure"
    assert position_1["passes_conservation_mask"] is False


def test_phase1_with_conservation_profile_reaches_mask_gate(tmp_path: Path) -> None:
    materialize_structure_authority(repo_root=repo_root(), output_root=tmp_path)
    materialize_contact_profile(repo_root=repo_root(), output_root=tmp_path)
    materialize_conservation_profile(
        repo_root=repo_root(),
        output_root=tmp_path,
        alignment_root=_write_alignment_inputs(tmp_path),
    )

    report = validate_checked_in_contracts(repo_root=repo_root(), phase="phase1_thread_contract", output_root=tmp_path)

    assert report.passed is False
    check_ids = {issue.check_id for issue in report.issues}
    assert "eco1_rt.evidence.conservation_profile_not_materialized" not in check_ids
    assert "eco1_rt.evidence.conservation_profile_metadata_mismatch" not in check_ids
    assert "eco1_rt.mask.mask_set_not_materialized" in check_ids


def test_phase1_rejects_conservation_profile_source_hash_mismatch(tmp_path: Path) -> None:
    materialize_structure_authority(repo_root=repo_root(), output_root=tmp_path)
    materialize_contact_profile(repo_root=repo_root(), output_root=tmp_path)
    result = materialize_conservation_profile(
        repo_root=repo_root(),
        output_root=tmp_path,
        alignment_root=_write_alignment_inputs(tmp_path),
    )
    table = pq.read_table(result.conservation_profile_path)
    rows = table.to_pylist()
    for row in rows:
        if row["profile_id"] == "broad_tao_homolog_rt":
            row["source_hash"] = "sha256:not-the-broad-alignment"
    pq.write_table(table.from_pylist(rows, schema=table.schema), result.conservation_profile_path)

    report = validate_checked_in_contracts(repo_root=repo_root(), phase="phase1_thread_contract", output_root=tmp_path)

    assert report.passed is False
    assert "eco1_rt.evidence.conservation_profile_source_hash_mismatch" in {issue.check_id for issue in report.issues}


def test_conservation_materializer_rejects_mismatched_target_row(tmp_path: Path) -> None:
    materialize_structure_authority(repo_root=repo_root(), output_root=tmp_path)
    alignment_root = _write_alignment_inputs(tmp_path, target_substitution=(301, "A"))

    with pytest.raises(ValueError, match="target row"):
        materialize_conservation_profile(
            repo_root=repo_root(),
            output_root=tmp_path,
            alignment_root=alignment_root,
        )


def _write_alignment_inputs(
    tmp_path: Path,
    *,
    target_substitution: tuple[int, str] | None = None,
) -> Path:
    residue_rows = pq.read_table(tmp_path / "residue_map.parquet").to_pylist()
    target = "".join(str(row["wt_aa"]) for row in residue_rows)
    if target_substitution is not None:
        position, aa = target_substitution
        target = target[: position - 1] + aa + target[position:]

    root = tmp_path / "alignments"
    root.mkdir()
    for profile_id in ("broad_tao_homolog_rt", "eco1_like_retron_rt"):
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
