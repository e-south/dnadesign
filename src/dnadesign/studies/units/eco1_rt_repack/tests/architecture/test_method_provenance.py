"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/architecture/test_method_provenance.py

Method-provenance documentation regression tests for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.studies.units.eco1_rt_repack.tests._helpers import repo_root


def _text(path: str) -> str:
    return (repo_root() / path).read_text(encoding="utf-8")


def test_primary_literature_roles_stay_separated_in_method_docs() -> None:
    fixed_backbone = _text("docs/studies/eco1_rt_repack/contexts/fixed-backbone-method.md")
    msa_method = _text("docs/studies/eco1_rt_repack/contexts/msa-method.md")
    mask_policy = _text("docs/studies/eco1_rt_repack/contexts/residue-mask-policy.md")

    assert "Tao et al. provides the fixed-backbone RT redesign pattern" in fixed_backbone
    assert "Mestre S1 is the accession and classification authority" in msa_method
    assert "Simon et al. provides RT-region and motif annotation grammar" in fixed_backbone
    assert "Wang and 7V9U define retained DNA/RNA geometry, direct contacts" in mask_policy
    assert "whole-database census alignment" in msa_method


def test_rt_interval_authority_is_not_documented_as_pending_after_audit() -> None:
    thread_spec = _text("docs/dev/plans/cross-tool/thread/2026-06-19-eco1-rt-repack-thread.md")
    msa_method = _text("docs/studies/eco1_rt_repack/contexts/msa-method.md")

    assert "Broader RT1-RT7 interval boxes remain deferred" not in thread_spec
    assert "RT1-RT7 intervals are annotation labels, not protection rules" in thread_spec
    assert "until manual motif authority and side-chain/contact-density evidence are materialized" not in msa_method
    assert "eco1_rt_clade9_plurality25_direct_contact5a_v1" in msa_method


def test_contact_risk_profile_is_documented_as_prior_evidence_only() -> None:
    mask_policy = " ".join(_text("docs/studies/eco1_rt_repack/contexts/residue-mask-policy.md").split())
    assert "contact-risk plots" in mask_policy
    assert "They do not change fixed or open positions" in mask_policy
    assert "at or below 5 A from retained DNA/RNA" in mask_policy
