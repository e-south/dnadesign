"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/twist_handoff/test_materialization.py

Eco1 RT Twist handoff materialization tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from Bio import SeqIO

from dnadesign.permuter import default_codon_table_path
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.twist_handoff import (
    materialize_twist_handoff,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.twist_handoff._fixtures import (
    REPO_ROOT,
    WT_GENBANK,
    write_twist_handoff_inputs,
)


def test_materializes_quote_ready_full_cds_handoff(tmp_path: Path) -> None:
    inputs = write_twist_handoff_inputs(tmp_path / "inputs")
    output_root = tmp_path / "out"
    stale_path = output_root / "genbank/stale_comparison_row.gb"
    stale_path.parent.mkdir(parents=True)
    stale_path.write_text("stale", encoding="utf-8")
    result = materialize_twist_handoff(repo_root=REPO_ROOT, output_root=output_root, **inputs)

    manifest = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    assert manifest["sequence_status"] == "quote_and_upload_ready"
    assert manifest["cloning_status"] == "blocked_pending_assembly_flanks_and_vendor_portal_complexity_check"
    assert len(manifest["sequences"]) == 8
    assert [row["selection_rank"] for row in manifest["sequences"]] == list(range(1, 9))
    assert manifest["selected_panel_source_count"] == 8
    assert manifest["selected_panel_composition"] == {
        "distal_scaffold_repack_v1": 2,
        "near_dna_rna_acid_free_v1": 3,
        "combined_near_acid_free_plus_distal_v1": 3,
    }
    assert manifest["assembly_state_scope"]["rt_msdna_oligomeric_state"] == "not_established"
    assert manifest["assay_host"] == {
        "species": "Escherichia coli",
        "strain": "K-12 MG1655",
        "ncbi_taxonomy_id": 511145,
    }
    assert manifest["codon_policy"]["policy_kind"] == "minimal_variant_aware_recoding"
    assert manifest["codon_policy"]["optimization_scope"] == "substituted_residues_only"
    assert manifest["codon_policy"]["global_codon_optimization"] is False
    assert manifest["codon_policy"]["codon_table"] == (
        "src/dnadesign/permuter/src/resources/codon_tables/codon_ecoli.csv"
    )
    assert manifest["codon_policy"]["codon_table_source"] == {
        "name": "Kazusa Codon Usage Database",
        "url": "https://www.kazusa.or.jp/codon/cgi-bin/showcodon.cgi?species=83333",
        "reference_organism": "Escherichia coli K-12",
        "reference_ncbi_taxonomy_id": 83333,
        "reference_cds_count": 14,
        "reference_codon_count": 5122,
    }
    assert manifest["codon_policy"]["host_suitability_scope"] == ("k12_lineage_reference_for_substituted_residues_only")
    assert set(manifest["input_hashes"]) == {
        "candidate_pool",
        "candidate_selection_panel",
        "codon_table",
        "foldcheck_fasta",
        "generation_policy_positions",
        "wild_type_cds_genbank",
    }
    for row in manifest["sequences"]:
        assert row["length_bp"] == 963
        assert row["translation_length_aa"] == 320
        assert row["canonical_protein_sha256"].startswith("sha256:")
        assert row["full_cds_sha256"].startswith("sha256:")
        assert row["wang_alpha1_r13_review_status"] in {"retained_wt", "substituted"}
        assert isinstance(row["wang_alpha1_mutation_count"], int)
        assert row["wang_alpha1_f10_substitution"] == "WT" or row["wang_alpha1_f10_substitution"].startswith("F10")
        assert row["wang_alpha1_r13_substitution"] == "WT" or row["wang_alpha1_r13_substitution"].startswith("R13")
        assert row["wang_r13a_interface_disruption_evidence_match"] is False
        assert row["rt_msdna_oligomeric_state_review_status"] == "not_established"
        assert set(row["qc"]) == {
            "gc_fraction",
            "gc_50bp_min_fraction",
            "gc_50bp_max_fraction",
            "gc_50bp_span_fraction",
            "max_homopolymer_run",
            "repeated_20mer_count",
            "forbidden_site_count",
        }
        assert row["qc"]["gc_50bp_min_fraction"] <= row["qc"]["gc_50bp_max_fraction"]
        assert row["qc"]["gc_50bp_span_fraction"] <= 0.5
        assert row["qc"]["forbidden_site_count"] == 0

    fasta = list(SeqIO.parse(result.fasta_path, "fasta"))
    assert len(fasta) == 8
    assert all(len(record.seq) == 963 for record in fasta)
    with result.twist_csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["name"] for row in rows] == [record.id for record in fasta]
    assert [row["sequence"] for row in rows] == [str(record.seq) for record in fasta]

    wt_record = SeqIO.read(
        REPO_ROOT / WT_GENBANK,
        "genbank",
    )
    wt_protein = str(wt_record.seq.translate())[:-1]
    best_codons: dict[str, tuple[float, str]] = {}
    with default_codon_table_path("ecoli").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["amino_acid"] != "*":
                choice = (float(row["frequency"]), row["codon"])
                best_codons[row["amino_acid"]] = max(best_codons.get(row["amino_acid"], choice), choice)
    foldcheck = {record.id: str(record.seq) for record in SeqIO.parse(inputs["foldcheck_fasta_path"], "fasta")}
    for dna_record, manifest_row in zip(fasta, manifest["sequences"], strict=True):
        protein = foldcheck[manifest_row["candidate_id"]]
        dna = str(dna_record.seq)
        for index, (wt_aa, candidate_aa) in enumerate(zip(wt_protein, protein, strict=True)):
            codon = dna[index * 3 : index * 3 + 3]
            if wt_aa == candidate_aa:
                assert codon == str(wt_record.seq[index * 3 : index * 3 + 3]).upper()
            else:
                assert codon == best_codons[candidate_aa][1]
        assert dna[-3:] == str(wt_record.seq[-3:]).upper()

    assert len(result.genbank_paths) == 8
    assert not stale_path.exists()
    for path in result.genbank_paths:
        record = SeqIO.read(path, "genbank")
        assert len(record.seq) == 963
        assert len([feature for feature in record.features if feature.type == "CDS"]) == 1
        mutation_features = [feature for feature in record.features if feature.type == "variation"]
        manifest_row = next(row for row in manifest["sequences"] if row["sequence_id"] == record.id)
        assert len(mutation_features) == len(manifest_row["mutation_tokens"])
        assert all(len(feature.location) == 3 for feature in mutation_features)
        assert any(feature.type == "misc_feature" for feature in record.features)
        feature_labels = {
            str(feature.qualifiers.get("label", [""])[0])
            for feature in record.features
            if feature.type == "misc_feature"
        }
        assert "wang_alpha1_interface_review" in feature_labels
        assert "wang_alpha1_R13_review" in feature_labels
        feature_notes = [str(note) for feature in record.features for note in feature.qualifiers.get("note", [])]
        assert any("rt_msdna_oligomeric_state=not_established" in note for note in feature_notes)


def test_fails_fast_when_panel_hash_does_not_match_foldcheck(tmp_path: Path) -> None:
    inputs = write_twist_handoff_inputs(tmp_path / "inputs")
    source = inputs["candidate_selection_panel_path"]
    table = pq.read_table(source)
    rows = table.to_pylist()
    rows[0]["sequence_hash"] = "sha256:" + "0" * 64
    bad_panel = tmp_path / "bad-panel.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), bad_panel)

    with pytest.raises(ValueError, match="panel sequence_hash"):
        materialize_twist_handoff(
            repo_root=REPO_ROOT,
            output_root=tmp_path / "out",
            candidate_selection_panel_path=bad_panel,
            candidate_pool_path=inputs["candidate_pool_path"],
            foldcheck_fasta_path=inputs["foldcheck_fasta_path"],
            generation_policy_positions_path=inputs["generation_policy_positions_path"],
            wild_type_genbank_path=inputs["wild_type_genbank_path"],
        )
