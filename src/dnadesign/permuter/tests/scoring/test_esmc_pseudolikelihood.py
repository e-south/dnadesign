"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/permuter/tests/scoring/test_esmc_pseudolikelihood.py

Tests for ESMC leave-one-out pseudo-likelihood helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from dnadesign.permuter.src.scoring.esmc_masked_marginal.contracts import CANONICAL_AMINO_ACIDS
from dnadesign.permuter.src.scoring.esmc_pseudolikelihood import (
    build_pseudolikelihood_jobs,
    build_sequence_pseudolikelihood_rows,
    normalize_pseudolikelihood_response,
    validate_pseudolikelihood_artifacts,
    write_pseudolikelihood_artifacts,
)


def test_build_pseudolikelihood_jobs_masks_each_selected_position() -> None:
    jobs = build_pseudolikelihood_jobs(sequence_id="wt", sequence="ACD", positions=(1, 3))

    assert len(jobs) == 2
    assert jobs[0].sequence_id == "wt"
    assert jobs[0].canonical_position == 1
    assert jobs[0].residue == "A"
    assert jobs[0].masked_sequence == "_CD"
    assert jobs[1].canonical_position == 3
    assert jobs[1].residue == "D"
    assert jobs[1].masked_sequence == "AC_"


def test_normalize_pseudolikelihood_response_uses_observed_residue_log_probability() -> None:
    job = build_pseudolikelihood_jobs(sequence_id="wt", sequence="ACD", positions=(2,))[0]

    rows = normalize_pseudolikelihood_response(
        job=job,
        logits_response={
            "logits": {
                "sequence": [_flat_logits(), _flat_logits(), _favored_logits("C"), _flat_logits(), _flat_logits()]
            }
        },
        aa_token_indices=_token_map(),
        model="esmc-test",
        biohub_request_hash="sha256:" + "1" * 64,
        biohub_query_hash="sha256:" + "2" * 64,
        retrieved_at="2026-07-02T00:00:00Z",
    )

    row = rows.position_row
    assert row["sequence_id"] == "wt"
    assert row["canonical_position"] == 2
    assert row["residue"] == "C"
    assert row["residue_log_probability"] > -1.0
    assert row["status"] == "accepted"
    assert row["raw_logits_response_hash"].startswith("sha256:")


def test_sequence_summary_computes_pseudolikelihood_and_wt_delta_control(tmp_path: Path) -> None:
    position_rows = [
        _position_row("wild_type", 1, -1.0),
        _position_row("wild_type", 2, -2.0),
        _position_row("variant_a", 1, -1.5),
        _position_row("variant_a", 2, -2.5),
    ]
    sequence_rows = build_sequence_pseudolikelihood_rows(
        position_rows=position_rows,
        expected_lengths_by_sequence_id={"wild_type": 2, "variant_a": 2},
        wt_sequence_id="wild_type",
    )

    by_id = {row["sequence_id"]: row for row in sequence_rows}
    assert by_id["wild_type"]["pll_total"] == -3.0
    assert by_id["wild_type"]["delta_pll_total_vs_wt"] == 0.0
    assert by_id["wild_type"]["delta_pll_mean_vs_wt"] == 0.0
    assert by_id["variant_a"]["pll_total"] == -4.0
    assert by_id["variant_a"]["delta_pll_total_vs_wt"] == -1.0
    assert by_id["variant_a"]["delta_pll_mean_vs_wt"] == -0.5

    artifacts = write_pseudolikelihood_artifacts(
        output_root=tmp_path,
        position_rows=position_rows,
        sequence_rows=sequence_rows,
        manifest={"schema_id": "test.pseudolikelihood", "authorization": "<redacted>"},
        request_hash="sha256:" + "1" * 64,
    )
    issues = validate_pseudolikelihood_artifacts(
        artifacts=artifacts,
        expected_sequence_count=2,
        request_hash="sha256:" + "1" * 64,
    )

    assert issues == []
    assert pq.read_table(artifacts.position_pll_path).num_rows == 4
    assert pq.read_table(artifacts.sequence_pll_path).num_rows == 2
    assert "authorization: <redacted>" in artifacts.manifest_path.read_text(encoding="utf-8")


def test_sequence_summary_keeps_partial_capped_runs_explicit() -> None:
    sequence_rows = build_sequence_pseudolikelihood_rows(
        position_rows=[_position_row("wild_type", 1, -1.0)],
        expected_lengths_by_sequence_id={"wild_type": 2},
        wt_sequence_id="wild_type",
    )

    assert sequence_rows[0]["status"] == "partial"
    assert sequence_rows[0]["pll_total"] is None
    assert sequence_rows[0]["delta_pll_mean_vs_wt"] is None


def _token_map() -> dict[str, int]:
    return {aa: index for index, aa in enumerate(CANONICAL_AMINO_ACIDS)}


def _flat_logits() -> list[float]:
    return [0.0 for _aa in CANONICAL_AMINO_ACIDS]


def _favored_logits(aa: str) -> list[float]:
    values = _flat_logits()
    values[_token_map()[aa]] = 3.0
    return values


def _position_row(sequence_id: str, position: int, residue_log_probability: float) -> dict[str, object]:
    return {
        "sequence_id": sequence_id,
        "sequence_hash": "sha256:" + sequence_id[-1:] * 64,
        "model": "esmc-test",
        "scoring_method_id": "esmc_leave_one_out_pseudolikelihood_v1",
        "biohub_request_hash": "sha256:" + "1" * 64,
        "biohub_query_hash": "sha256:" + str(position) * 64,
        "canonical_position": position,
        "residue_index_zero_based": position - 1,
        "residue": "A",
        "masked_sequence_hash": "sha256:" + "2" * 64,
        "token_count": 4,
        "vocab_size": 20,
        "logit_residue_offset": 1,
        "residue_log_probability": residue_log_probability,
        "raw_logits_response_hash": "sha256:" + "3" * 64,
        "retrieved_at": "2026-07-02T00:00:00Z",
        "status": "accepted",
        "failure_reason": "",
    }
