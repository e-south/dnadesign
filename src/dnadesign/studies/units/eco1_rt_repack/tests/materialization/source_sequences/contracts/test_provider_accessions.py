"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/source_sequences/contracts/test_provider_accessions.py

Provider accession contract tests for Eco1 conservation source sequences.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.contracts import (
    ProviderAccessionPolicy,
    parse_conservation_source_contract,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.io import (
    load_yaml_mapping,
)
from dnadesign.studies.units.eco1_rt_repack.tests._helpers import repo_root


def test_provider_accession_policy_comes_from_conservation_source_contract() -> None:
    contract = parse_conservation_source_contract(_contract_payload())
    policy = ProviderAccessionPolicy.from_contract(contract)

    assert policy.provider_for_accession("WP_099010551.1") == "ncbi_protein_efetch"
    assert policy.provider_for_accession("EIJ70524.1") == "ncbi_protein_efetch"
    assert policy.provider_for_accession("fig|511145.12.peg.42") == "bv_brc_feature_protein_fasta"
    assert policy.valid_provider_accession("ncbi_protein_efetch", "WP_BROAD_1") is False


def test_provider_accession_contract_rejects_missing_required_patterns() -> None:
    payload = _contract_payload()
    for provider in payload["sequence_providers"]:
        if provider["id"] == "bv_brc_feature_protein_fasta":
            provider.pop("accession_patterns")

    with pytest.raises(ValueError, match="accession_patterns"):
        parse_conservation_source_contract(payload)


def _contract_payload() -> dict[str, object]:
    path = repo_root() / "docs/studies/eco1_rt_repack/workbench/provenance/conservation-sources.yaml"
    return dict(load_yaml_mapping(path))
