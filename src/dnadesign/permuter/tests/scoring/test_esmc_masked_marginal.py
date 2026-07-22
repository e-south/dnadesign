"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/permuter/tests/scoring/test_esmc_masked_marginal.py

Tests for ESMC masked-marginal protein DMS scoring helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pyarrow.parquet as pq

from dnadesign.permuter import ProteinDmsRequest
from dnadesign.permuter.src.scoring.esmc_masked_marginal import (
    CANONICAL_AMINO_ACIDS,
    build_masked_marginal_jobs,
    normalize_masked_marginal_response,
    render_masked_marginal_plots,
    validate_masked_marginal_artifacts,
    write_masked_marginal_artifacts,
)


def test_build_masked_marginal_jobs_uses_protein_dms_position_semantics() -> None:
    request = ProteinDmsRequest(ref_name="wt", sequence="ACD", positions=(2,))

    jobs = build_masked_marginal_jobs(request)

    assert len(jobs) == 1
    assert jobs[0].sequence_id == "wt"
    assert jobs[0].canonical_position == 2
    assert jobs[0].residue_index_zero_based == 1
    assert jobs[0].wt_aa == "C"
    assert jobs[0].masked_sequence == "A_D"
    assert jobs[0].sequence_hash.startswith("sha256:")
    assert jobs[0].masked_sequence_hash.startswith("sha256:")


def test_normalize_masked_marginal_response_computes_position_and_substitution_rows() -> None:
    job = build_masked_marginal_jobs(ProteinDmsRequest(ref_name="wt", sequence="ACD", positions=(2,)))[0]
    token_map = _token_map()

    rows = normalize_masked_marginal_response(
        job=job,
        logits_response={"logits": {"sequence": [_flat_logits(), _favored_logits("W"), _flat_logits()]}},
        aa_token_indices=token_map,
        model="esmc-test",
        biohub_request_hash="sha256:" + "1" * 64,
        biohub_query_hash="sha256:" + "2" * 64,
        retrieved_at="2026-06-27T00:00:00Z",
    )

    assert rows.position_row["canonical_position"] == 2
    assert rows.position_row["wt_aa"] == "C"
    assert rows.position_row["best_alt_aa"] == "W"
    assert "fraction_negative_llr" not in rows.position_row
    assert rows.position_row["fraction_negative_alternate_llr"] < 1.0
    assert len(rows.substitution_rows) == 19
    assert {row["alt_aa"] for row in rows.substitution_rows} == set(CANONICAL_AMINO_ACIDS) - {"C"}
    best = max(rows.substitution_rows, key=lambda row: float(row["llr"]))
    assert best["alt_aa"] == "W"
    assert best["llr"] > 0.0


def test_write_validate_and_plot_masked_marginal_artifacts(tmp_path: Path) -> None:
    jobs = build_masked_marginal_jobs(ProteinDmsRequest(ref_name="wt", sequence="ACD", positions=(1, 2)))
    token_map = _token_map()
    position_rows = []
    substitution_rows = []
    for job in jobs:
        normalized = normalize_masked_marginal_response(
            job=job,
            logits_response={"logits": {"sequence": [_favored_logits("A"), _favored_logits("W"), _flat_logits()]}},
            aa_token_indices=token_map,
            model="esmc-test",
            biohub_request_hash="sha256:" + "1" * 64,
            biohub_query_hash="sha256:" + str(job.canonical_position) * 64,
            retrieved_at="2026-06-27T00:00:00Z",
        )
        position_rows.append(normalized.position_row)
        substitution_rows.extend(normalized.substitution_rows)

    artifacts = write_masked_marginal_artifacts(
        output_root=tmp_path,
        position_rows=position_rows,
        substitution_rows=substitution_rows,
        manifest={"schema_id": "test.manifest", "authorization": "<redacted>"},
        request_hash="sha256:" + "1" * 64,
    )
    issues = validate_masked_marginal_artifacts(
        artifacts=artifacts,
        expected_position_count=2,
        request_hash="sha256:" + "1" * 64,
    )
    plot_artifacts = render_masked_marginal_plots(
        position_entropy_path=artifacts.position_entropy_path,
        substitution_llr_path=artifacts.substitution_llr_path,
        output_root=tmp_path / "plots",
        file_prefix="wt_",
        position_context_spans=[
            {
                "start": 1,
                "end": 2,
                "label": "RT interval",
                "annotate_label": "RT1",
                "color": "#111111",
                "alpha": 0.05,
            },
            {
                "start": 2,
                "end": 2,
                "label": "Motif anchor",
                "annotate_label": "NAxxH",
                "color": "#D55E00",
                "alpha": 0.15,
            },
        ],
    )

    assert issues == []
    assert pq.read_table(artifacts.position_entropy_path).num_rows == 2
    assert pq.read_table(artifacts.substitution_llr_path).num_rows == 38
    assert plot_artifacts.entropy_by_position_path.stat().st_size > 0
    assert plot_artifacts.substitution_llr_heatmap_path.stat().st_size > 0
    for path in (
        plot_artifacts.entropy_by_position_path,
        plot_artifacts.fraction_negative_alternate_llr_path,
        plot_artifacts.substitution_llr_heatmap_path,
    ):
        svg_text = path.read_text(encoding="utf-8")
        svg_root = ET.parse(path).getroot()
        assert "<title" in svg_text
        assert "<desc" in svg_text
        assert svg_root.attrib["role"] == "img"
    entropy_text = plot_artifacts.entropy_by_position_path.read_text(encoding="utf-8")
    fraction_text = plot_artifacts.fraction_negative_alternate_llr_path.read_text(encoding="utf-8")
    heatmap_text = plot_artifacts.substitution_llr_heatmap_path.read_text(encoding="utf-8")
    assert "WT residue" not in entropy_text
    assert "WT residue" not in fraction_text
    assert "WT residue" not in heatmap_text
    assert ">WT<" not in heatmap_text
    assert "LLR vs WT" in heatmap_text
    assert "Ec86 position" in heatmap_text
    assert "RT interval" in entropy_text
    assert "RT1" in entropy_text
    assert "Motif anchor" in fraction_text
    assert "NAxxH" in fraction_text
    assert "authorization: <redacted>" in artifacts.manifest_path.read_text(encoding="utf-8")


def _token_map() -> dict[str, int]:
    return {aa: index for index, aa in enumerate(CANONICAL_AMINO_ACIDS)}


def _flat_logits() -> list[float]:
    return [0.0 for _aa in CANONICAL_AMINO_ACIDS]


def _favored_logits(aa: str) -> list[float]:
    values = _flat_logits()
    values[_token_map()[aa]] = 3.0
    return values
