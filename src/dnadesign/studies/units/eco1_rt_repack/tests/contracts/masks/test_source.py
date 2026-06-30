"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/contracts/masks/test_source.py

Manual mask-authority source contract tests for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.masks import (
    load_manual_mask_authority_source,
    wang_direct_contact_prior_positions_from_source,
)
from dnadesign.studies.units.eco1_rt_repack.tests._helpers import repo_root


def test_manual_mask_source_declares_wang_direct_contact_priors_and_audited_rt_intervals() -> None:
    authority_source = load_manual_mask_authority_source(repo_root())

    source_basis_ids = {source["id"] for source in authority_source["source_basis"]}
    assert "wang_et_al_2022_ec86_cryoem_structure_priors" in source_basis_ids
    direct_contact_positions = wang_direct_contact_prior_positions_from_source(authority_source)
    assert direct_contact_positions == {49, 51, 55, 56, 73, 231, 257, 264}
    assert 13 not in direct_contact_positions
