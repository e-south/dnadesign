"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/tests/adapters/colabfold/test_outputs.py

ColabFold output normalization tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.thread.adapters.colabfold.index import ColabFoldOutputIndex
from dnadesign.thread.adapters.colabfold.outputs import build_colabfold_foldcheck_rows
from dnadesign.thread.foldcheck.hashes import sequence_hash


def test_colabfold_output_parser_emits_accepted_rows_with_wt_baseline_rmsd(tmp_path: Path) -> None:
    output_root = tmp_path / "colabfold_outputs"
    output_root.mkdir()
    _write_ca_pdb(output_root / "wild_type_unrelaxed_rank_001_alphafold2_model_1_seed_000.pdb", bfactor=91.0)
    _write_ca_pdb(
        output_root / "thread_candidate_a_unrelaxed_rank_001_alphafold2_model_1_seed_000.pdb",
        bfactor=84.0,
        y_offset=0.2,
    )
    manifest = _request_manifest(["wild_type", "thread_candidate_a"])

    rows = build_colabfold_foldcheck_rows(
        output_root=output_root,
        request_manifest=manifest,
        runtime_version="colabfold-test",
        runtime_parameters={"command": "colabfold_batch", "mode": "smoke"},
    )

    assert [row["candidate_id"] for row in rows] == ["wild_type", "thread_candidate_a"]
    assert {row["status"] for row in rows} == {"accepted"}
    assert rows[0]["wt_baseline_artifact_id"] == "self"
    assert rows[0]["backbone_rmsd_to_reference"] == 0.0
    assert rows[0]["plddt"] == 91.0
    assert rows[1]["wt_baseline_artifact_id"] == "wild_type"
    assert rows[1]["plddt"] == 84.0
    assert rows[1]["backbone_rmsd_to_reference"] > 0.0
    assert rows[1]["runtime_parameters_hash"].startswith("sha256:")


def test_colabfold_output_parser_turns_missing_candidate_output_into_failure_row(tmp_path: Path) -> None:
    output_root = tmp_path / "colabfold_outputs"
    output_root.mkdir()
    _write_ca_pdb(output_root / "wild_type_unrelaxed_rank_001_alphafold2_model_1_seed_000.pdb", bfactor=91.0)
    manifest = _request_manifest(["wild_type", "thread_candidate_missing"])

    rows = build_colabfold_foldcheck_rows(
        output_root=output_root,
        request_manifest=manifest,
        runtime_version="colabfold-test",
        runtime_parameters={"command": "colabfold_batch", "mode": "smoke"},
    )

    missing = rows[1]
    assert missing["candidate_id"] == "thread_candidate_missing"
    assert missing["status"] == "errored"
    assert missing["missing_metric_reason"] == "colabfold_output_missing"
    assert missing["rejection_reason"] == "colabfold_output_missing"


def test_colabfold_output_parser_ignores_numeric_candidate_id_when_selecting_rank(tmp_path: Path) -> None:
    output_root = tmp_path / "colabfold_outputs"
    output_root.mkdir()
    candidate_id = "thread_candidate_318124686032"
    _write_ca_pdb(output_root / "wild_type_unrelaxed_rank_001_alphafold2_model_1_seed_000.pdb", bfactor=91.0)
    _write_ca_pdb(
        output_root / f"{candidate_id}_unrelaxed_rank_001_alphafold2_model_1_seed_000.pdb",
        bfactor=84.0,
        y_offset=0.2,
    )
    (output_root / f"{candidate_id}_predicted_aligned_error_v1.json").write_text(
        '{"predicted_aligned_error": [[9.0]]}',
        encoding="utf-8",
    )
    rank_score = output_root / f"{candidate_id}_scores_rank_001_alphafold2_model_1_seed_000.json"
    rank_score.write_text('{"pae": [[1.0]]}', encoding="utf-8")
    manifest = _request_manifest(["wild_type", candidate_id])

    rows = build_colabfold_foldcheck_rows(
        output_root=output_root,
        request_manifest=manifest,
        runtime_version="colabfold-test",
        runtime_parameters={"command": "colabfold_batch", "mode": "smoke"},
    )

    assert rows[1]["candidate_id"] == candidate_id
    assert rows[1]["status"] == "accepted"
    assert rows[1]["score_artifact_path"].endswith(rank_score.name)
    assert rows[1]["pae_summary"]["mean"] == 1.0


def test_colabfold_output_parser_does_not_steal_prefix_candidate_outputs(tmp_path: Path) -> None:
    output_root = tmp_path / "colabfold_outputs"
    output_root.mkdir()
    shorter_id = "thread_candidate_a"
    longer_id = "thread_candidate_a_extra"
    _write_ca_pdb(output_root / "wild_type_unrelaxed_rank_001_alphafold2_model_1_seed_000.pdb", bfactor=91.0)
    _write_ca_pdb(
        output_root / f"{longer_id}_unrelaxed_rank_001_alphafold2_model_1_seed_000.pdb",
        bfactor=84.0,
        y_offset=0.2,
    )
    manifest = _request_manifest(["wild_type", shorter_id, longer_id])

    rows = build_colabfold_foldcheck_rows(
        output_root=output_root,
        request_manifest=manifest,
        runtime_version="colabfold-test",
        runtime_parameters={"command": "colabfold_batch", "mode": "smoke"},
    )

    assert rows[1]["candidate_id"] == shorter_id
    assert rows[1]["status"] == "errored"
    assert rows[1]["missing_metric_reason"] == "colabfold_output_missing"
    assert rows[2]["candidate_id"] == longer_id
    assert rows[2]["status"] == "accepted"


def test_colabfold_output_parser_requires_colabfold_suffix_boundary_for_prefixed_ids(tmp_path: Path) -> None:
    output_root = tmp_path / "colabfold_outputs"
    output_root.mkdir()
    shorter_id = "seq1"
    longer_id = "seq1_variant"
    _write_ca_pdb(output_root / "wild_type_unrelaxed_rank_001_alphafold2_model_1_seed_000.pdb", bfactor=91.0)
    _write_ca_pdb(
        output_root / f"{longer_id}_unrelaxed_rank_001_alphafold2_model_1_seed_000.pdb",
        bfactor=84.0,
        y_offset=0.2,
    )
    manifest = _request_manifest(["wild_type", shorter_id, longer_id])

    rows = build_colabfold_foldcheck_rows(
        output_root=output_root,
        request_manifest=manifest,
        runtime_version="colabfold-test",
        runtime_parameters={"command": "colabfold_batch", "mode": "smoke"},
    )

    assert rows[1]["candidate_id"] == shorter_id
    assert rows[1]["status"] == "errored"
    assert rows[1]["missing_metric_reason"] == "colabfold_output_missing"
    assert rows[2]["candidate_id"] == longer_id
    assert rows[2]["status"] == "accepted"
    output_index = ColabFoldOutputIndex.from_output_root(output_root, sequence_ids=[shorter_id, longer_id])
    assert output_index.select_model_pdb(shorter_id) is None
    assert output_index.select_model_pdb(longer_id) is not None


def _request_manifest(sequence_ids: list[str]) -> dict[str, object]:
    return {
        "request_hash": "sha256:" + "1" * 64,
        "runtime_kind": "alphafold_family_colabfold",
        "wt_sequence_id": "wild_type",
        "reference_structure_id": "ec86kit_reference_fixture",
        "threshold_policy_id": "foldcheck_thresholds_fixture",
        "threshold_values": {"requires_wt_baseline": True},
        "sequences": [
            {
                "sequence_id": sequence_id,
                "sequence_hash": sequence_hash("A" * 4),
                "source_kind": "wild_type_baseline" if sequence_id == "wild_type" else "proteinmpnn_candidate",
                "length": 4,
            }
            for sequence_id in sequence_ids
        ],
    }


def _write_ca_pdb(path: Path, *, bfactor: float, y_offset: float = 0.0) -> None:
    bend = 0.4 if y_offset == 0.0 else 0.8
    coords = [(0.0, y_offset, 0.0), (1.0, y_offset, 0.0), (2.0, y_offset, 0.0), (3.0, y_offset + bend, 0.0)]
    lines = []
    for index, (x_coord, y_coord, z_coord) in enumerate(coords, start=1):
        lines.append(
            f"ATOM  {index:5d}  CA  ALA A{index:4d}    "
            f"{x_coord:8.3f}{y_coord:8.3f}{z_coord:8.3f}  1.00{bfactor:6.2f}           C"
        )
    path.write_text("\n".join(lines) + "\nEND\n", encoding="utf-8")
