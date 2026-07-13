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

from dnadesign.studies.units.eco1_rt_repack.tests.materialization.selection_readiness._candidate_fixtures import (
    candidate_rows,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.selection_readiness._local_structure_fixtures import (
    write_local_structure_inputs,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.selection_readiness._source_fixtures import (
    write_selection_source_inputs,
)


def write_inputs(class_root: Path, source_root: Path) -> dict[str, list[dict[str, object]]]:
    class_root.mkdir(parents=True)
    candidates = candidate_rows()
    _write_parquet(class_root / "candidate_pool.parquet", candidates)
    _write_foldcheck_input_sequences(class_root / "foldcheck_request/input_sequences.fasta", candidates)
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
    write_local_structure_inputs(class_root, candidates)
    write_selection_source_inputs(source_root)
    return {"candidate_pool": candidates}


def _write_foldcheck_input_sequences(path: Path, candidates: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [("wild_type", "A" * 320)]
    records.extend((str(row["candidate_id"]), "AA" + str(row["sequence"]) + "A" * 9) for row in candidates)
    path.write_text(
        "".join(f">{candidate_id}\n{protein_sequence}\n" for candidate_id, protein_sequence in records),
        encoding="utf-8",
    )


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
