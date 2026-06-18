"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/permuter/tests/api/test_public_api.py

Public API behavior contracts for in-memory Permuter workflows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import replace

import pandas as pd
import pytest

from dnadesign.permuter import (
    CodingDnaDmsRequest,
    CodingDnaDmsVariantMetadata,
    EvaluatorPlan,
    InferFeatureRequest,
    InferFeatureSourceDataset,
    InferSequenceViewSelector,
    MetricSpec,
    NucleotideDmsRequest,
    ProteinDmsRequest,
    default_codon_table_path,
    evaluate_variants,
    generate_variants,
    infer_feature_request_from_mapping,
    materialize_result,
    read_infer_feature_request_manifest,
    validate_dataset,
    write_infer_feature_request_manifest,
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


def test_public_api_coding_dna_dms_defaults_to_full_coding_sequence(tmp_path) -> None:
    table = tmp_path / "codons.csv"
    table.write_text(CODON_CSV, encoding="utf-8")

    result = generate_variants(
        CodingDnaDmsRequest(
            ref_name="toy_rt_cds",
            sequence="AAAAAG",
            codon_table=table,
            metadata={"study_id": "unit-study"},
        )
    )

    assert len(result.records) == 4
    assert result.metadata["permuter"]["positions"] == (1, 2)
    assert result.metadata["permuter"]["excluded_codon_positions"] == ()
    assert {record.metadata["permuter"]["aa_pos"] for record in result.records} == {1, 2}
    assert {record.metadata["permuter"]["codon_index"] for record in result.records} == {0, 1}


def test_public_api_coding_dna_dms_default_excludes_terminal_stop_codon(tmp_path) -> None:
    table = tmp_path / "codons.csv"
    table.write_text(CODON_CSV, encoding="utf-8")

    result = generate_variants(
        CodingDnaDmsRequest(
            ref_name="toy_rt_cds",
            sequence="AAATAA",
            codon_table=table,
        )
    )

    assert len(result.records) == 2
    assert result.metadata["permuter"]["positions"] == (1,)
    assert result.metadata["permuter"]["excluded_codon_positions"] == (2,)
    assert {record.metadata["permuter"]["codon_wt"] for record in result.records} == {"AAA"}


def test_public_api_coding_dna_dms_default_rejects_internal_stop_codon(tmp_path) -> None:
    table = tmp_path / "codons.csv"
    table.write_text(CODON_CSV, encoding="utf-8")

    with pytest.raises(ValueError, match="internal stop codon"):
        generate_variants(
            CodingDnaDmsRequest(
                ref_name="toy_rt_cds",
                sequence="AAATAAAAA",
                codon_table=table,
            )
        )


def test_public_api_coding_dna_dms_default_rejects_unsupported_reference_codon(tmp_path) -> None:
    table = tmp_path / "codons.csv"
    table.write_text(CODON_CSV, encoding="utf-8")

    with pytest.raises(ValueError, match="ATG at AA position 1"):
        generate_variants(
            CodingDnaDmsRequest(
                ref_name="toy_rt_cds",
                sequence="ATGAAA",
                codon_table=table,
            )
        )


def test_public_api_coding_dna_dms_is_exhaustive_over_sense_codons_with_top_ecoli_codons() -> None:
    codon_table = default_codon_table_path("ecoli")

    result = generate_variants(
        CodingDnaDmsRequest(
            ref_name="toy_rt_cds",
            sequence="ATGGCT",
            codon_table=codon_table,
        )
    )

    protein_alphabet = set("ACDEFGHIKLMNPQRSTVWY")
    stop_codons = {"TAA", "TAG", "TGA"}
    with codon_table.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["amino_acid"] != "*"]
    top_codon_by_aa = {
        aa: max(
            (row for row in rows if row["amino_acid"] == aa),
            key=lambda row: float(row["frequency"]),
        )["codon"]
        for aa in protein_alphabet
    }

    assert len(result.records) == 38
    assert result.metadata["permuter"]["positions"] == (1, 2)
    assert result.metadata["permuter"]["excluded_codon_positions"] == ()
    records_by_pos = {}
    for record in result.records:
        meta = CodingDnaDmsVariantMetadata.from_record(record)
        records_by_pos.setdefault(meta.aa_pos, []).append((record, meta))
        codons = {record.sequence[index : index + 3] for index in range(0, len(record.sequence), 3)}
        assert not (codons & stop_codons)
        assert meta.aa_alt != "*"
        assert meta.codon_new == top_codon_by_aa[meta.aa_alt]
        assert meta.codon_new not in stop_codons

    assert {meta.aa_wt for _record, meta in records_by_pos[1]} == {"M"}
    assert {meta.aa_alt for _record, meta in records_by_pos[1]} == protein_alphabet - {"M"}
    assert {meta.aa_wt for _record, meta in records_by_pos[2]} == {"A"}
    assert {meta.aa_alt for _record, meta in records_by_pos[2]} == protein_alphabet - {"A"}


def test_public_api_coding_dna_dms_rejects_explicit_stop_codon_position(tmp_path) -> None:
    table = tmp_path / "codons.csv"
    table.write_text(CODON_CSV, encoding="utf-8")

    with pytest.raises(ValueError, match="TAA at AA position 2"):
        generate_variants(
            CodingDnaDmsRequest(
                ref_name="toy_rt_cds",
                sequence="AAATAA",
                codon_table=table,
                positions=(2,),
            )
        )


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


def test_public_api_writes_non_executing_infer_feature_request_manifest(tmp_path) -> None:
    request = InferFeatureRequest(
        source_dataset=InferFeatureSourceDataset(
            usr_root="workspaces/studies/example/usr",
            dataset_id="example_construct_contexts_v1",
        ),
        feature_bundle_ref="docs/studies/example/fixtures/infer/evo2-feature-bundle.yaml",
        sequence_view_selectors=(
            InferSequenceViewSelector(view_name="dual_cassette_2000bp_seq_mean"),
            InferSequenceViewSelector(alias="rt_cds_bidir_anchor_mean"),
        ),
        requested_outputs=("log_likelihood", "output_layer_mean", "intermediate_embedding"),
    )

    manifest_path = write_infer_feature_request_manifest(request, tmp_path / "infer-handoff.yaml")
    parsed = read_infer_feature_request_manifest(manifest_path)

    assert parsed == request
    assert parsed.to_mapping() == {
        "kind": "permuter_infer_feature_request_v1",
        "source_owner": "permuter",
        "execution_owner": "infer",
        "writeback_owner": "infer",
        "source_dataset": {
            "usr_root": "workspaces/studies/example/usr",
            "dataset_id": "example_construct_contexts_v1",
        },
        "feature_bundle_ref": "docs/studies/example/fixtures/infer/evo2-feature-bundle.yaml",
        "sequence_view_selectors": [
            {"view_name": "dual_cassette_2000bp_seq_mean"},
            {"alias": "rt_cds_bidir_anchor_mean"},
        ],
        "requested_outputs": ["log_likelihood", "output_layer_mean", "intermediate_embedding"],
    }


def test_public_api_infer_feature_request_rejects_broad_view_selectors() -> None:
    payload = {
        "kind": "permuter_infer_feature_request_v1",
        "source_owner": "permuter",
        "execution_owner": "infer",
        "writeback_owner": "infer",
        "source_dataset": {
            "usr_root": "workspaces/studies/example/usr",
            "dataset_id": "example_construct_contexts_v1",
        },
        "feature_bundle_ref": "docs/studies/example/fixtures/infer/evo2-feature-bundle.yaml",
        "sequence_view_selectors": [
            {
                "product_kind": "realized_context",
                "orientation": "forward",
            }
        ],
        "requested_outputs": ["log_likelihood"],
    }

    with pytest.raises(ValueError, match="explicit view_name or alias"):
        infer_feature_request_from_mapping(payload)


def test_public_api_infer_feature_request_allows_construct_owned_source_dataset() -> None:
    request = infer_feature_request_from_mapping(
        {
            "kind": "permuter_infer_feature_request_v1",
            "source_owner": "construct",
            "execution_owner": "infer",
            "writeback_owner": "infer",
            "source_dataset": {
                "usr_root": "workspaces/studies/example/usr",
                "dataset_id": "example_construct_contexts_v1",
            },
            "feature_bundle_ref": "docs/studies/example/fixtures/infer/evo2-feature-bundle.yaml",
            "sequence_view_selectors": [{"view_name": "explicit_construct_view"}],
            "requested_outputs": ["log_likelihood"],
        }
    )

    assert request.source_owner == "construct"


def test_public_api_infer_feature_request_rejects_unknown_source_owner() -> None:
    payload = {
        "kind": "permuter_infer_feature_request_v1",
        "source_owner": "opal",
        "execution_owner": "infer",
        "writeback_owner": "infer",
        "source_dataset": {
            "usr_root": "workspaces/studies/example/usr",
            "dataset_id": "example_construct_contexts_v1",
        },
        "feature_bundle_ref": "docs/studies/example/fixtures/infer/evo2-feature-bundle.yaml",
        "sequence_view_selectors": [{"view_name": "explicit_construct_view"}],
        "requested_outputs": ["log_likelihood"],
    }

    with pytest.raises(ValueError, match="source_owner must be one of"):
        infer_feature_request_from_mapping(payload)


def test_public_api_infer_feature_request_keeps_infer_as_execution_owner() -> None:
    with pytest.raises(ValueError, match="execution_owner"):
        InferFeatureRequest(
            source_dataset=InferFeatureSourceDataset(
                usr_root="workspaces/studies/example/usr",
                dataset_id="example_construct_contexts_v1",
            ),
            feature_bundle_ref="docs/studies/example/fixtures/infer/evo2-feature-bundle.yaml",
            sequence_view_selectors=(InferSequenceViewSelector(view_name="one_explicit_view"),),
            requested_outputs=("log_likelihood",),
            execution_owner="permuter",  # type: ignore[arg-type]
        )


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
