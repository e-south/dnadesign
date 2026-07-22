"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/foldcheck_review/fixtures.py

Fixtures for Eco1 fold-check review materialization tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from dnadesign.thread.candidates.proteinmpnn import write_candidate_table
from dnadesign.thread.foldcheck import sequence_hash, write_foldcheck_report


def write_review_inputs(output_root: Path, *, local_model_paths: bool) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    _write_residue_map(output_root / "residue_map.parquet")
    reference_path = output_root / "proteinmpnn_request" / "chain_a_backbone.pdb"
    _write_reference_pdb(reference_path)
    _write_foldcheck_request(output_root / "foldcheck_request")

    candidates = [
        ("thread_candidate_best_rmsd", 0.8, 92.4, 5.6, 78, 1.1, 303, 0.1),
        ("thread_candidate_best_plddt", 1.3, 92.9, 5.4, 81, 1.3, 303, 0.3),
        ("thread_candidate_high_mutation", 1.4, 90.1, 6.8, 90, 1.2, 101, 0.3),
        ("thread_candidate_worst_rmsd", 28.9, 91.0, 6.4, 79, 1.1, 202, 0.1),
        ("thread_candidate_low_plddt", 1.5, 88.6, 7.3, 82, 1.3, 202, 0.3),
        ("thread_candidate_intermediate", 3.7, 90.4, 6.6, 83, 1.3, 202, 0.3),
    ]
    _write_candidate_table(output_root / "candidate_table.parquet", candidates)
    model_root = output_root / "colabfold_models"
    model_root.mkdir()
    fold_rows = [_accepted_foldcheck_row("wild_type", _sequence("wild_type"), model_root, local_model_paths)]
    for candidate_id, rmsd, plddt, pae_mean, *_ in candidates:
        fold_rows.append(
            _accepted_foldcheck_row(
                candidate_id,
                _sequence(candidate_id),
                model_root,
                local_model_paths,
                rmsd=rmsd,
                plddt=plddt,
                pae_mean=pae_mean,
            )
        )
    write_foldcheck_report(output_root / "foldcheck_report.parquet", fold_rows, request_hash="sha256:" + "8" * 64)


def _write_residue_map(path: Path) -> None:
    rows = [
        {
            "canonical_position": position,
            "wt_aa": "A",
            "mapping_status": "mapped" if 3 <= position <= 311 else "unresolved_structure",
        }
        for position in range(1, 321)
    ]
    pq.write_table(pa.Table.from_pylist(rows), path)


def _write_foldcheck_request(request_root: Path) -> None:
    request_root.mkdir(parents=True, exist_ok=True)
    sequences = [{"sequence_id": "wild_type", "sequence_hash": sequence_hash(_sequence("wild_type"))}]
    for candidate_id in (
        "thread_candidate_best_rmsd",
        "thread_candidate_best_plddt",
        "thread_candidate_high_mutation",
        "thread_candidate_worst_rmsd",
        "thread_candidate_low_plddt",
        "thread_candidate_intermediate",
    ):
        sequences.append({"sequence_id": candidate_id, "sequence_hash": sequence_hash(_sequence(candidate_id))})
    request_root.joinpath("foldcheck_request_manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_id": "thread.foldcheck_request",
                "request_hash": "sha256:" + "8" * 64,
                "sequence_count": len(sequences),
                "wt_sequence_id": "wild_type",
                "sequences": sequences,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _write_candidate_table(
    path: Path,
    candidates: list[tuple[str, float, float, float, int, float, int, float]],
) -> None:
    rows: list[dict[str, Any]] = []
    for rank, (candidate_id, _rmsd, _plddt, _pae, mutation_count, score, seed, temperature) in enumerate(
        candidates,
        start=1,
    ):
        rows.append(
            {
                "candidate_id": candidate_id,
                "source_sample_id": f"sample-{rank}",
                "backend_run_id": "proteinmpnn-fixture",
                "request_hash": "sha256:" + "7" * 64,
                "sequence_hash": sequence_hash(_sequence(candidate_id)),
                "sequence": "A" * 309,
                "score": score,
                "global_score": score + 1.0,
                "seq_recovery": 0.5,
                "seed": seed,
                "temperature": temperature,
                "sample_index": rank,
                "duplicate_sample_count": 1,
                "mutation_count": mutation_count,
                "mutable_mutation_count": mutation_count,
                "protected_mutation_count": 0,
                "outside_mutable_positions": [],
                "canonical_mutations": [f"A{position}G" for position in range(3, 3 + mutation_count)],
                "status": "accepted",
                "rank": rank,
            }
        )
    write_candidate_table(path, rows, request_hash="sha256:" + "7" * 64)


def _accepted_foldcheck_row(
    candidate_id: str,
    sequence: str,
    model_root: Path,
    local_model_paths: bool,
    *,
    rmsd: float = 0.0,
    plddt: float = 93.0,
    pae_mean: float = 5.0,
) -> dict[str, Any]:
    model_path = model_root / f"{candidate_id}_unrelaxed_rank_001_alphafold2_model_1_seed_000.pdb"
    _write_full_sequence_pdb(model_path, y_offset=rmsd / 100.0, bfactor=plddt)
    source_path = (
        model_path if local_model_paths else Path("/project/dunlop/esouth/foldcheck/fixture") / model_path.name
    )
    return {
        "candidate_id": candidate_id,
        "runtime_kind": "alphafold_family_colabfold",
        "runtime_version": "colabfold-test",
        "input_sequence_hash": sequence_hash(sequence),
        "reference_structure_id": "ec86kit_7v9u_protomer1",
        "wt_baseline_artifact_id": "self" if candidate_id == "wild_type" else "wild_type",
        "runtime_parameters_hash": "sha256:" + "6" * 64,
        "threshold_id": "eco1_rt_foldcheck_thresholds_v1",
        "threshold_values": {"requires_wt_baseline": True},
        "plddt": plddt,
        "pae_summary": {"status": "parsed", "mean": pae_mean, "max": 30.0},
        "backbone_rmsd_to_reference": rmsd,
        "protected_contact_retention": None,
        "status": "accepted",
        "rejection_reason": "",
        "missing_metric_reason": "",
        "model_artifact_path": str(source_path),
        "score_artifact_path": "",
    }


def _write_reference_pdb(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_ca_pdb(path, residue_count=309, y_offset=0.0, bfactor=0.0)


def _write_full_sequence_pdb(path: Path, *, y_offset: float, bfactor: float) -> None:
    _write_ca_pdb(path, residue_count=320, y_offset=y_offset, bfactor=bfactor)


def _write_ca_pdb(path: Path, *, residue_count: int, y_offset: float, bfactor: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for atom_index, residue_position in enumerate(range(1, residue_count + 1), start=1):
        x_coord = float(residue_position)
        lines.append(
            f"ATOM  {atom_index:5d}  CA  ALA A{residue_position:4d}    "
            f"{x_coord:8.3f}{y_offset:8.3f}{0.0:8.3f}  1.00{bfactor:6.2f}           C"
        )
    path.write_text("\n".join(lines) + "\nEND\n", encoding="utf-8")


def _sequence(candidate_id: str) -> str:
    if candidate_id == "wild_type":
        return "A" * 320
    return "A" * 319 + "G"
