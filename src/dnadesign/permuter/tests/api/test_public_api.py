"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/tests/api/test_public_api.py

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pandas as pd
import pytest

from dnadesign.permuter import (
    CodingDnaDmsRequest,
    CodingDnaDmsVariantMetadata,
    EvaluatorPlan,
    MetricSpec,
    NucleotideDmsRequest,
    ProteinDmsRequest,
    default_codon_table_path,
    evaluate_variants,
    generate_variants,
    materialize_result,
    validate_dataset,
)

CODON_CSV = """codon,amino_acid,fraction,frequency
AAA,K,0.73,33.2
AAG,K,0.27,12.1
AAC,N,0.53,24.4
AAT,N,0.47,21.9
CAA,Q,0.30,12.1
CAG,Q,0.70,27.7
"""


def test_public_api_generates_nucleotide_dms_without_filesystem() -> None:
    result = generate_variants(
        NucleotideDmsRequest(
            ref_name="toy_dna",
            sequence="AC",
            metadata={"study": "unit"},
        )
    )

    assert result.bio_type == "dna"
    assert len(result.records) == 6
    assert {record.ref_name for record in result.records} == {"toy_dna"}
    assert result.metadata["study"] == "unit"
    assert result.metadata["permuter"]["protocol"] == "nucleotide_dms"
    assert all(record.metadata["study"] == "unit" for record in result.records)
    assert {record.metadata["permuter"]["protocol"] for record in result.records} == {"nucleotide_dms"}
    assert all(record.sequence != "AC" for record in result.records)


def test_public_api_generates_protein_dms_for_selected_positions() -> None:
    result = generate_variants(
        ProteinDmsRequest(
            ref_name="toy_protein",
            sequence="MA",
            positions=(2,),
            metadata={"caller": "study-runtime"},
        )
    )

    assert result.bio_type == "protein"
    assert len(result.records) == 19
    assert result.metadata["caller"] == "study-runtime"
    assert result.metadata["permuter"]["protocol"] == "protein_dms"
    assert {record.metadata["caller"] for record in result.records} == {"study-runtime"}
    assert {record.metadata["permuter"]["aa_pos"] for record in result.records} == {2}
    assert all(record.modifications == (f"aa pos=2 wt=A alt={record.sequence[1]}",) for record in result.records)


def test_public_api_fails_fast_on_invalid_dna_sequence() -> None:
    with pytest.raises(ValueError, match="DNA"):
        generate_variants(NucleotideDmsRequest(ref_name="bad", sequence="AX"))


def test_public_api_generates_coding_dna_dms_with_explicit_codon_policy(tmp_path) -> None:
    table = tmp_path / "codons.csv"
    table.write_text(CODON_CSV, encoding="utf-8")

    result = generate_variants(
        CodingDnaDmsRequest(
            ref_name="toy_rt_cds",
            sequence="AAA",
            codon_table=table,
            positions=(1,),
            metadata={"study_id": "unit-study"},
        )
    )

    assert result.bio_type == "dna"
    assert len(result.records) == 2
    assert result.metadata["study_id"] == "unit-study"
    assert result.metadata["permuter"]["protocol"] == "coding_dna_dms"
    assert {record.sequence for record in result.records} == {"AAC", "CAG"}
    assert {record.metadata["study_id"] for record in result.records} == {"unit-study"}
    assert {record.metadata["permuter"]["aa_alt"] for record in result.records} == {"N", "Q"}
    assert all(record.metadata["permuter"]["codon_policy"] == "top" for record in result.records)
    assert {CodingDnaDmsVariantMetadata.from_record(record).aa_alt for record in result.records} == {"N", "Q"}
    assert all(CodingDnaDmsVariantMetadata.from_record(record).codon_wt == "AAA" for record in result.records)


def test_public_api_materializes_coding_dna_dms_with_canonical_usr_ids_and_var_ids(tmp_path) -> None:
    table = tmp_path / "codons.csv"
    table.write_text(CODON_CSV, encoding="utf-8")
    result = generate_variants(
        CodingDnaDmsRequest(
            ref_name="toy_rt_cds",
            sequence="AAA",
            codon_table=table,
            positions=(1,),
        )
    )

    dataset = materialize_result(result, tmp_path / "coding_dna_dataset")
    report = validate_dataset(dataset, strict=True)
    df = pd.read_parquet(dataset.records_path)
    record_ids_by_sequence = {record.sequence: record.id for record in result.records}

    assert report.ok is True
    assert "permuter__variant_id" not in df.columns
    assert set(df["id"]) == {hashlib.sha1(f"dna|{sequence}".encode("utf-8")).hexdigest() for sequence in df["sequence"]}
    assert {
        row.sequence: row.permuter__var_id for row in df[["sequence", "permuter__var_id"]].itertuples(index=False)
    } == record_ids_by_sequence


def test_public_api_materialize_rejects_conflicting_variant_id_metadata(tmp_path) -> None:
    generated = generate_variants(NucleotideDmsRequest(ref_name="toy_dna", sequence="AC"))
    record = generated.records[0]
    conflicting_record = replace(
        record,
        metadata={
            **record.metadata,
            "permuter": {
                **record.metadata["permuter"],
                "var_id": "not-the-record-id",
            },
        },
    )
    conflicting_result = replace(generated, records=(conflicting_record,))

    with pytest.raises(ValueError, match="metadata.permuter.var_id conflicts"):
        materialize_result(conflicting_result, tmp_path / "conflicting")


def test_public_api_materialize_rejects_variant_id_metadata_alias(tmp_path) -> None:
    generated = generate_variants(NucleotideDmsRequest(ref_name="toy_dna", sequence="AC"))
    record = generated.records[0]
    aliased_record = replace(
        record,
        metadata={
            **record.metadata,
            "permuter": {
                **record.metadata["permuter"],
                "variant_id": record.id,
            },
        },
    )
    aliased_result = replace(generated, records=(aliased_record,))

    with pytest.raises(ValueError, match="metadata.permuter.variant_id is not supported"):
        materialize_result(aliased_result, tmp_path / "aliased")


def test_public_api_strict_validation_rejects_variant_id_column_alias(tmp_path) -> None:
    generated = generate_variants(NucleotideDmsRequest(ref_name="toy_dna", sequence="AC"))
    dataset = materialize_result(generated, tmp_path / "dataset")
    df = pd.read_parquet(dataset.records_path)
    df["permuter__variant_id"] = df["permuter__var_id"]
    df.to_parquet(dataset.records_path, index=False)

    with pytest.raises(ValueError, match="permuter__variant_id"):
        validate_dataset(dataset, strict=True)


def test_public_api_rejects_coding_dna_dms_without_codon_table() -> None:
    with pytest.raises(ValueError, match="codon_table is required"):
        generate_variants(CodingDnaDmsRequest(ref_name="bad", sequence="AAA", codon_table=""))


def test_public_api_rejects_coding_dna_dms_above_max_variants(tmp_path) -> None:
    table = tmp_path / "codons.csv"
    table.write_text(CODON_CSV, encoding="utf-8")

    with pytest.raises(ValueError, match="above max_variants=1"):
        generate_variants(
            CodingDnaDmsRequest(
                ref_name="toy_rt_cds",
                sequence="AAA",
                codon_table=table,
                positions=(1,),
                max_variants=1,
            )
        )


def test_public_api_reserves_permuter_metadata_key() -> None:
    with pytest.raises(ValueError, match="reserved"):
        generate_variants(
            ProteinDmsRequest(
                ref_name="bad_metadata",
                sequence="MA",
                metadata={"permuter": {"caller": "not allowed"}},
            )
        )


def test_coding_dna_dms_metadata_accessor_rejects_non_coding_dna_records() -> None:
    result = generate_variants(ProteinDmsRequest(ref_name="toy_protein", sequence="MA", positions=(2,)))

    with pytest.raises(ValueError, match="coding_dna_dms"):
        CodingDnaDmsVariantMetadata.from_record(result.records[0])


def test_default_codon_table_path_is_public_and_checked() -> None:
    path = default_codon_table_path("ecoli")

    assert path.name == "codon_ecoli.csv"
    assert path.exists()


def test_public_api_scores_materializes_and_validates_dataset(tmp_path) -> None:
    generated = generate_variants(NucleotideDmsRequest(ref_name="toy_dna", sequence="AC"))
    scored = evaluate_variants(
        generated,
        EvaluatorPlan(
            metrics=(
                MetricSpec(
                    id="smoke",
                    evaluator="placeholder",
                    metric="log_likelihood",
                ),
            )
        ),
    )

    first_record = scored.records[0]
    assert isinstance(first_record.metadata["permuter"]["observed"]["smoke"], float)

    dataset = materialize_result(scored, tmp_path / "toy_dataset")
    report = validate_dataset(dataset, strict=True)
    df = pd.read_parquet(dataset.records_path)

    assert dataset.row_count == 6
    assert report.ok is True
    assert report.metric_ids == ("smoke",)
    assert "permuter__observed__smoke" in df.columns
    assert set(df["permuter__ref"]) == {"toy_dna"}
    assert dataset.ref_path and dataset.ref_path.exists()


def test_public_api_refuses_metric_overwrite_without_contract() -> None:
    generated = generate_variants(NucleotideDmsRequest(ref_name="toy_dna", sequence="AC"))
    plan = EvaluatorPlan(
        metrics=(
            MetricSpec(
                id="smoke",
                evaluator="placeholder",
                metric="log_likelihood",
            ),
        )
    )
    scored = evaluate_variants(generated, plan)

    with pytest.raises(ValueError, match="already exists"):
        evaluate_variants(scored, plan)
