"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/tests/test_rt_lnrna_sponging_construct_triage_variant_genbank_catalog.py

Variant GenBank catalog checks for the RT-lnRNA sponging construct triage study.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

from dnadesign.studies.studies.rt_lnrna_sponging_construct_triage.variant_genbank_catalog import (
    build_variant_genbank_catalog,
)


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


def test_variant_genbank_catalog_extracts_slot_authority_and_preserves_metadata() -> None:
    catalog = build_variant_genbank_catalog(repo_root=_repo_root())

    assert catalog.ok, "\n".join(catalog.errors)
    assert catalog.variant_count == 36

    retron26 = catalog.record("retron26")
    assert retron26.plasmid_name == "pES-retron-26"
    assert not Path(retron26.source_path).is_absolute()
    assert retron26.source_path.startswith("docs/studies/rt_lnrna_sponging_construct_triage/")
    assert retron26.lnrna.span_0 == (186, 359)
    assert retron26.lnrna.length_nt == 173
    assert retron26.rt_cds.span_0 == (524, 1487)
    assert retron26.rt_cds.length_nt == 963
    assert retron26.rt_cds.authority_kind == "wt_eco1_rt"
    assert retron26.construct_projection_status == "representable"
    assert 'P2 loop edited to "GUU"' in retron26.comment
    assert retron26.benchling_url.startswith("https://benchling.com/s/seq-U5fZ9LdotIsqORxKR2Tb")

    retron43 = catalog.record("retron43")
    assert retron43.lnrna.length_nt == 187
    assert retron43.rt_cds.sequence_sha256 == retron26.rt_cds.sequence_sha256
    assert retron43.construct_spans_0["lnrna"] == (123, 310)
    assert retron43.construct_spans_0["rt_cds"] == (475, 1438)

    retron47 = catalog.record("retron47")
    assert retron47.variant_class == "rt_translational_fusion"
    assert retron47.rt_cds.authority_kind == "rt_translational_fusion"
    assert retron47.rt_cds.span_0 == (524, 1694)
    assert retron47.rt_cds.length_nt == 1170
    assert retron47.construct_projection_status == "representable"
    assert retron47.construct_spans_0["lnrna"] == (27, 200)
    assert retron47.construct_spans_0["rt_cds"] == (365, 1535)
    assert "context_flanks_truncated_to_1600bp" in retron47.qc_flags

    retron48 = catalog.record("retron48")
    assert retron48.construct_projection_status == "representable"
    assert retron48.construct_spans_0["lnrna"] == (27, 200)
    assert retron48.construct_spans_0["rt_cds"] == (365, 1535)

    retron49 = catalog.record("retron49")
    assert retron49.variant_class == "rt_point_mutation"
    assert retron49.rt_cds.authority_kind == "rt_point_mutation"
    assert retron49.rt_cds.mutation_labels == ("R38Y",)
    assert retron49.rt_cds.sequence_sha256 != retron26.rt_cds.sequence_sha256
    assert retron49.construct_projection_status == "representable"

    retron170 = catalog.record("retron170")
    assert "WT tetO2 derived binding site" in retron170.comment
    assert retron170.reader_design_id == "pES-retron-170; pBbS2c-rfp"

    retron176 = catalog.record("retron176")
    assert retron176.lnrna.length_nt == 178
    assert retron176.rt_cds.length_nt == 963
    assert retron176.construct_projection_status == "representable"
    assert "033-GTG-ACG-CAC" in retron176.comment

    bl21 = catalog.record("msrmsdwt_bl21")
    assert bl21.variant_class == "native_lnrna_wt_rt"
    assert bl21.lnrna.label == "record"
    assert bl21.lnrna.span_0 == (0, 170)
    assert bl21.lnrna.length_nt == 170
    assert bl21.rt_cds.authority_kind == "wt_eco1_rt"
    assert bl21.construct_candidate_id == "rt_lnrna_pair__eco1_wt_rt__msrmsdwt_bl21_lnrna__native"


def test_variant_genbank_catalog_source_files_are_study_owned_and_complete() -> None:
    repo_root = _repo_root()
    catalog = build_variant_genbank_catalog(repo_root=repo_root)
    genbank_dir = repo_root / "docs/studies/rt_lnrna_sponging_construct_triage/workbench/provenance/genbank"
    temp_dir = repo_root.parent / "temp_location_for_retron_genbanks"

    assert not temp_dir.exists()
    assert len(list(genbank_dir.glob("*.gb"))) == 40
    assert not catalog.missing_metadata_source_files
    assert not catalog.missing_genbank_source_files

    catalog_path = (
        repo_root
        / "docs/studies/rt_lnrna_sponging_construct_triage/workbench/provenance/retron-variant-genbank-catalog.yaml"
    )
    payload = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    assert payload["catalog_id"] == "rt_lnrna_sponging_construct_triage_retron_variant_genbank_catalog_v1"
    assert payload["variant_count"] == 36
    assert payload["records"]["retron47"]["rt_cds"]["length_nt"] == 1170
    assert payload["records"]["msrmsdwt_bl21"]["lnrna"]["length_nt"] == 170
