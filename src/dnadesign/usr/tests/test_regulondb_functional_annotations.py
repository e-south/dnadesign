"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_regulondb_functional_annotations.py

Regression tests for RegulonDB functional annotations USR.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dnadesign.usr.src.regulondb.functional_annotations import (
    build_regulondb_go_projection,
    write_regulondb_go_projection,
)


def _write_table(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    lines = ["\t".join(fields)]
    lines.extend("\t".join(str(row.get(field, "")) for field in fields) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _dataset_root(tmp_path: Path) -> Path:
    dataset = tmp_path / "usr_regulondb_native_promoters"
    _write_table(
        dataset / "records.parquet",
        [
            {"id": "usr-a", "bio_type": "dna", "sequence": "ACGT"},
            {"id": "usr-b", "bio_type": "dna", "sequence": "TGCA"},
        ],
    )
    _write_table(
        dataset / "_relations/regulatory_interactions.parquet",
        [
            {
                "usr_id": "usr-a",
                "promoter_id": "PM1",
                "regulator_id": "R1",
                "regulator_name": "H-NS",
                "regulatory_interaction_id": "RI1",
            },
            {
                "usr_id": "usr-a",
                "promoter_id": "PM1",
                "regulator_id": "R1",
                "regulator_name": "H-+NS",
                "regulatory_interaction_id": "RI2",
            },
            {
                "usr_id": "usr-b",
                "promoter_id": "PM2",
                "regulator_id": "R2",
                "regulator_name": "LexA",
                "regulatory_interaction_id": "RI3",
            },
        ],
    )
    return dataset


def _terms_root(tmp_path: Path) -> tuple[Path, Path]:
    terms = tmp_path / "regulator_go_terms.tsv"
    coverage = tmp_path / "regulator_go_coverage.tsv"
    _write_tsv(
        terms,
        [
            {
                "regulator_id": "R1",
                "regulator_name": "H-NS",
                "regulator_gene_name": "hns",
                "gene_symbol": "hns",
                "go_aspect": "molecular_function",
                "go_id": "GO:0001217",
                "go_name": "DNA-binding transcription repressor activity",
                "go_namespace": "molecular_function",
                "source_column": "GO terms (molecular function)",
                "source_route": "biocyc_smarttable_gene_go_terms",
                "identity_source_id": "regulondb_13_network_regulator_gene",
                "biocyc_kb_version": "29.6",
                "smarttable_id": "biocyc-test",
            },
            {
                "regulator_id": "R2",
                "regulator_name": "LexA",
                "regulator_gene_name": "lexA",
                "gene_symbol": "lexA",
                "go_aspect": "biological_process",
                "go_id": "GO:0009432",
                "go_name": "SOS response",
                "go_namespace": "biological_process",
                "source_column": "GO terms (biological process)",
                "source_route": "biocyc_smarttable_gene_go_terms",
                "identity_source_id": "regulondb_13_network_regulator_gene",
                "biocyc_kb_version": "29.6",
                "smarttable_id": "biocyc-test",
            },
        ],
    )
    _write_tsv(
        coverage,
        [
            {
                "regulator_id": "R1",
                "regulator_name": "H-NS",
                "regulator_gene_name": "hns",
                "gene_symbol_count": "1",
                "matched_go_term_count": "1",
                "mapping_status": "matched",
                "identity_source_id": "regulondb_13_network_regulator_gene",
                "biocyc_kb_version": "29.6",
                "smarttable_id": "biocyc-test",
            },
            {
                "regulator_id": "R1",
                "regulator_name": "H-+NS",
                "regulator_gene_name": "hns",
                "gene_symbol_count": "1",
                "matched_go_term_count": "0",
                "mapping_status": "unmatched_gene_symbol",
                "identity_source_id": "regulondb_13_network_regulator_gene",
                "biocyc_kb_version": "29.6",
                "smarttable_id": "biocyc-test",
            },
            {
                "regulator_id": "R2",
                "regulator_name": "LexA",
                "regulator_gene_name": "lexA",
                "gene_symbol_count": "1",
                "matched_go_term_count": "1",
                "mapping_status": "matched",
                "identity_source_id": "regulondb_13_network_regulator_gene",
                "biocyc_kb_version": "29.6",
                "smarttable_id": "biocyc-test",
            },
        ],
    )
    return terms, coverage


def test_regulondb_go_projection_collapses_duplicate_regulator_aliases(
    tmp_path: Path,
) -> None:
    dataset = _dataset_root(tmp_path)
    terms, coverage = _terms_root(tmp_path)

    projection = build_regulondb_go_projection(
        dataset_root=dataset,
        terms_path=terms,
        coverage_path=coverage,
        min_covered_regulator_fraction=1.0,
    )

    assert projection.summary["interacting_regulator_count"] == 2
    assert projection.summary["covered_regulator_count"] == 2
    assert projection.summary["promoter_regulator_go_term_rows"] == 2
    hns_rows = [row for row in projection.regulator_go_coverage_rows if row["regulator_id"] == "R1"]
    assert len(hns_rows) == 1
    assert hns_rows[0]["usr_promoter_count"] == 1
    assert hns_rows[0]["source_promoter_count"] == 1
    assert hns_rows[0]["regulatory_interaction_count"] == 2
    assert hns_rows[0]["matched_go_term_count"] == 1
    assert hns_rows[0]["mapping_status"] == "matched"


def test_regulondb_go_projection_keeps_regulators_with_blank_names(tmp_path: Path) -> None:
    dataset = _dataset_root(tmp_path)
    interactions_path = dataset / "_relations/regulatory_interactions.parquet"
    interaction_rows = pq.read_table(interactions_path).to_pylist()
    for row in interaction_rows:
        if row["regulator_id"] == "R2":
            row["regulator_name"] = ""
    _write_table(interactions_path, interaction_rows)
    terms, coverage = _terms_root(tmp_path)

    projection = build_regulondb_go_projection(
        dataset_root=dataset,
        terms_path=terms,
        coverage_path=coverage,
        min_covered_regulator_fraction=1.0,
    )

    assert projection.summary["interacting_regulator_count"] == 2
    lex_a_row = next(row for row in projection.regulator_go_coverage_rows if row["regulator_id"] == "R2")
    assert lex_a_row["regulator_name"] == "LexA"


def test_regulondb_go_projection_fails_when_coverage_is_too_sparse(
    tmp_path: Path,
) -> None:
    dataset = _dataset_root(tmp_path)
    terms, coverage = _terms_root(tmp_path)
    _write_tsv(
        terms,
        [
            {
                "regulator_id": "R1",
                "regulator_name": "H-NS",
                "regulator_gene_name": "hns",
                "gene_symbol": "hns",
                "go_aspect": "molecular_function",
                "go_id": "GO:0001217",
                "go_name": "DNA-binding transcription repressor activity",
                "go_namespace": "molecular_function",
                "source_column": "GO terms (molecular function)",
                "source_route": "biocyc_smarttable_gene_go_terms",
                "identity_source_id": "regulondb_13_network_regulator_gene",
                "biocyc_kb_version": "29.6",
                "smarttable_id": "biocyc-test",
            }
        ],
    )

    with pytest.raises(ValueError, match="BioCyc GO coverage below required minimum"):
        build_regulondb_go_projection(
            dataset_root=dataset,
            terms_path=terms,
            coverage_path=coverage,
            min_covered_regulator_fraction=1.0,
        )


def test_write_regulondb_go_projection_materializes_relation_sidecars(
    tmp_path: Path,
) -> None:
    dataset = _dataset_root(tmp_path)
    terms, coverage = _terms_root(tmp_path)
    projection = build_regulondb_go_projection(
        dataset_root=dataset,
        terms_path=terms,
        coverage_path=coverage,
    )

    write_regulondb_go_projection(projection, dataset_root=dataset)

    relations = dataset / "_relations"
    assert pq.read_table(relations / "regulator_go_terms.parquet").num_rows == 2
    assert pq.read_table(relations / "promoter_regulator_go_terms.parquet").num_rows == 2
    assert pq.read_table(relations / "regulator_go_coverage.parquet").num_rows == 2
