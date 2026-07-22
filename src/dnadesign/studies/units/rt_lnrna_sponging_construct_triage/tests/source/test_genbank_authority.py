"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/source/test_genbank_authority.py

GenBank source-authority checks for the RT-lnRNA sponging construct triage study.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.genbank_authority import (
    GenBankAuthorityRegistry,
    GenBankAuthoritySource,
    run_default_authority_audit,
    validate_genbank_authority_registry,
)


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


def _candidate_fixture(name: str) -> dict[str, object]:
    path = (
        _repo_root()
        / "docs"
        / "studies"
        / "rt_lnrna_sponging_construct_triage"
        / "operations"
        / "contract"
        / "fixtures"
        / "construct-subjects"
        / name
    )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_genbank_source_authority_audit_passes_for_registered_references() -> None:
    audit = run_default_authority_audit(repo_root=_repo_root())

    assert audit.ok, "\n".join(audit.errors)

    retron26_vector = audit.source("pes_retron_26_vector")
    retron43_vector = audit.source("pes_retron_43_vector")
    region = audit.source("dual_cassette_2000bp_region")
    eco1_rt = audit.source("retron_eco1_rt")
    orientation = audit.source("retron_179_orientation_reference")

    assert retron26_vector.record_id == "pES-retron-26"
    assert retron26_vector.length == 4956
    assert retron26_vector.topology == "circular"
    assert retron26_vector.feature("msr").span_1 == (195, 273)
    assert retron26_vector.feature("msd[tetO]").span_1 == (265, 338)
    assert retron26_vector.feature("WT loop").sequence == "GCCT"

    assert retron43_vector.record_id == "pES-retron-43"
    assert retron43_vector.length == 4970
    assert retron43_vector.feature("loop").sequence == "CGGG"
    assert retron43_vector.feature("msd[tetO]").span_1 == (265, 352)

    assert region.record_id == "2000bp-region"
    assert region.length == 2000
    assert region.topology == "linear"
    assert region.feature("a1(20)").span_1 == (131, 150)
    assert region.feature("ECD_00831").span_1 == (469, 1431)

    assert eco1_rt.length == 963
    assert eco1_rt.feature("ECD_00831").span_1 == (1, 963)

    assert orientation.feature("Right Base").sequence == "ATTG"
    assert orientation.feature("Cap").span_1 == (118, 120)
    assert orientation.feature("Foldback").sequence == "GAGTCTCTC"
    assert audit.rt_cds_identity_source_ids == (
        "pes_retron_26_vector",
        "pes_retron_43_vector",
        "retron_eco1_rt",
    )


def test_genbank_authority_validator_fails_missing_required_feature() -> None:
    registry = GenBankAuthorityRegistry(
        sources=(
            GenBankAuthoritySource(
                source_id="broken_retron26_vector",
                path=("docs/studies/rt_lnrna_sponging_construct_triage/workbench/provenance/genbank/pes-retron-26.gb"),
                role="test_only",
                required_unique_labels=("missing-msr",),
            ),
        ),
    )

    audit = validate_genbank_authority_registry(repo_root=_repo_root(), registry=registry)

    assert not audit.ok
    assert "broken_retron26_vector: missing required feature label 'missing-msr'" in audit.errors


def test_anchor_fixtures_are_backed_by_genbank_source_authority() -> None:
    retron26 = _candidate_fixture("retron26-working-anchor.yaml")
    retron43 = _candidate_fixture("retron43-failed-anchor.yaml")

    for fixture in (retron26, retron43):
        candidate = fixture["candidate"]
        msd_design_spec = fixture["msd_design_spec"]
        assert isinstance(candidate, dict)
        assert isinstance(msd_design_spec, dict)
        assert candidate["rt_cds_sequence_id"] == "genbank:retron-eco1-rt.gb#ECD_00831"
        assert candidate["rt_protein_provenance_id"] == "genbank:retron-eco1-rt.gb#CDS"
        assert candidate["source_authority_status"] == "resolved_by_genbank_source_authority"
        assert "exact_eco1_wt_rt_cds_source" not in candidate["blockers"]
        assert "exact_dual_cassette_plasmid_constants" not in candidate["blockers"]
        assert msd_design_spec["payload_sequence"] == "TCCCTATCAGTGATAGAGA"

    assert retron26["candidate"]["lnrna_sequence_id"] == "genbank:pes-retron-26-a1-a2.gb#a1-a2"
    assert retron26["msd_design_spec"]["left_base"] == "CCCG"
    assert retron26["msd_design_spec"]["right_base"] == "TCTG"
    assert retron26["msd_design_spec"]["snapback_source_id"] == "genbank:pes-retron-26-a1-a2.gb#WT-loop"

    assert retron43["candidate"]["lnrna_sequence_id"] == "genbank:pes-retron-43.gb#a1-a2"
    assert retron43["msd_design_spec"]["left_base"] == "CTTG"
    assert retron43["msd_design_spec"]["right_base"] == "TCGA"
    assert retron43["msd_design_spec"]["snapback_source_id"] == "genbank:pes-retron-43.gb#loop"
