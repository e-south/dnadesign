"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/selection_readiness/_fixtures.py

Panel-selection test fixtures for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes.specs import (
    ALL_SPECS,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.selection_readiness._source_fixtures import (
    write_selection_source_inputs,
)


def write_inputs(class_root: Path, source_root: Path) -> dict[str, list[dict[str, object]]]:
    class_root.mkdir(parents=True)
    candidates = candidate_rows()
    _write_parquet(class_root / "candidate_pool.parquet", candidates)
    _write_parquet(class_root / "foldcheck_report.parquet", _foldcheck_report_rows(candidates))
    _write_parquet(class_root / "foldcheck_review/foldcheck_candidate_ranking.parquet", _fold_review_rows(candidates))
    _write_parquet(
        class_root / "review_deliverables/biohub_esmc_sequence_scoring/biohub_esmc_variant_llr_scores.parquet",
        _llr_rows(candidates, model="esmc-300m-2024-12", offset=1.0),
    )
    _write_parquet(
        class_root
        / "review_deliverables/biohub_esmc_sequence_scoring/esmc_6b_2024_12/biohub_esmc_variant_llr_scores.parquet",
        _llr_rows(candidates, model="esmc-6b-2024-12", offset=-10.0),
    )
    _write_parquet(class_root / "biohub_esmc/sae_feature_window_summary.parquet", _sae_rows(candidates))
    _write_local_structure_inputs(class_root, candidates)
    write_selection_source_inputs(source_root)
    return {"candidate_pool": candidates}


def candidate_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, spec in enumerate(ALL_SPECS, start=1):
        rows.append(
            {
                "candidate_id": f"candidate_{index}",
                "sequence_hash": f"sha256:{index:064d}",
                "sequence": sequence(index),
                "status": "accepted",
                "rank": index,
                "design_class_id": spec.design_class_id,
                "mask_policy_id": spec.design_class_id,
                "seed": 101 + index,
                "temperature": 0.1 if index % 2 else 0.3,
                "mutation_count": 20 + index,
                "mutable_mutation_count": 20 + index,
                "protected_mutation_count": 0,
                "outside_mutable_positions": [],
                "canonical_mutations": [f"A{index + 2}G", f"L{index + 20}V"],
            }
        )
    rows.extend([_low_confidence_candidate(), _mask_blocked_candidate()])
    return rows


def sequence(offset: int) -> str:
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    return "".join(alphabet[(offset + i) % len(alphabet)] for i in range(64))


def _low_confidence_candidate() -> dict[str, object]:
    baseline_id = ALL_SPECS[0].design_class_id
    return {
        "candidate_id": "candidate_low_conf",
        "sequence_hash": "sha256:" + "a" * 64,
        "sequence": sequence(21),
        "status": "accepted",
        "rank": 999,
        "design_class_id": baseline_id,
        "mask_policy_id": baseline_id,
        "seed": 303,
        "temperature": 0.3,
        "mutation_count": 25,
        "mutable_mutation_count": 25,
        "protected_mutation_count": 0,
        "outside_mutable_positions": [],
        "canonical_mutations": ["A7G"],
    }


def _mask_blocked_candidate() -> dict[str, object]:
    baseline_id = ALL_SPECS[0].design_class_id
    return {
        "candidate_id": "candidate_blocked_by_mask",
        "sequence_hash": "sha256:" + "b" * 64,
        "sequence": sequence(22),
        "status": "accepted",
        "rank": 1000,
        "design_class_id": baseline_id,
        "mask_policy_id": baseline_id,
        "seed": 303,
        "temperature": 0.3,
        "mutation_count": 26,
        "mutable_mutation_count": 25,
        "protected_mutation_count": 1,
        "outside_mutable_positions": [198],
        "canonical_mutations": ["Y198F"],
    }


def _foldcheck_report_rows(candidates: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = [
        {
            "candidate_id": "wild_type",
            "input_sequence_hash": "sha256:" + "0" * 64,
            "status": "accepted",
            "plddt": 93.0,
        }
    ]
    rows.extend(
        {
            "candidate_id": row["candidate_id"],
            "input_sequence_hash": row["sequence_hash"],
            "status": "accepted",
            "plddt": 91.0,
        }
        for row in candidates
    )
    return rows


def _fold_review_rows(candidates: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, row in enumerate(candidates, start=1):
        review_class = "low_confidence" if row["candidate_id"] == "candidate_low_conf" else "strong_fold_preserved"
        rows.append(
            {
                "candidate_id": row["candidate_id"],
                "mutation_count": row["mutation_count"],
                "mutable_mutation_count": row["mutable_mutation_count"],
                "foldcheck_status": "accepted",
                "plddt": 94.0 - index / 10,
                "wt_runtime_ca_rmsd": 0.5 + index / 100,
                "cryoem_mapped_ca_rmsd": 2.0 + index / 100,
                "review_class": review_class,
                "review_rank": index,
            }
        )
    return rows


def _llr_rows(candidates: list[dict[str, object]], *, model: str, offset: float) -> list[dict[str, object]]:
    return [
        {
            "candidate_id": row["candidate_id"],
            "sequence_hash": row["sequence_hash"],
            "model": model,
            "llr_total": offset - float(index),
            "llr_per_mutation": (offset - float(index)) / float(row["mutation_count"]),
        }
        for index, row in enumerate(candidates)
    ]


def _sae_rows(candidates: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in candidates:
        for window_id in ("catalytic_palm_control", "thumb_palm_na_binding_surface", "mutable_annulus"):
            rows.append(
                {
                    "candidate_id": row["candidate_id"],
                    "sequence_hash": row["sequence_hash"],
                    "window_id": window_id,
                    "cosine_distance_to_wt": 0.0002,
                    "activation_delta_sum_vs_wt": 0.1,
                    "window_redundancy_rank": 1,
                    "used_for_selection": False,
                }
            )
    return rows


def _write_parquet(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _write_local_structure_inputs(class_root: Path, candidates: list[dict[str, object]]) -> None:
    structure_root = class_root / "foldcheck_review/structures"
    model_root = structure_root / "full_fold_set"
    model_root.mkdir(parents=True, exist_ok=True)
    rows = [(position, float(position), float(position % 17), float(position % 29)) for position in range(1, 321)]
    _write_ca_pdb(structure_root / "ec86kit_chain_a_backbone_reference.pdb", rows)
    for index, candidate in enumerate(candidates, start=1):
        shift = float(index) / 100.0
        _write_ca_pdb(
            model_root / f"{candidate['candidate_id']}.pdb",
            [(position, x + shift, y - shift, z + shift) for position, x, y, z in rows],
        )


def _write_ca_pdb(path: Path, rows: list[tuple[int, float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        (f"ATOM  {index:5d}  CA  ALA A{residue:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 90.00           C\n")
        for index, (residue, x, y, z) in enumerate(rows, start=1)
    ]
    path.write_text("".join(lines) + "END\n", encoding="utf-8")
