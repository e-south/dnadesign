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
    assert "Wang/Ec86 direct-contact priors must be explicit study-owned records" in mask_policy
    assert "whole-database census alignment" in msa_method


def test_rt_interval_authority_is_not_documented_as_pending_after_audit() -> None:
    thread_spec = _text("docs/dev/plans/cross-tool/thread/2026-06-19-eco1-rt-repack-thread.md")
    msa_method = _text("docs/studies/eco1_rt_repack/contexts/msa-method.md")

    assert "Broader RT1-RT7 interval boxes remain deferred" not in thread_spec
    assert "RT1-RT7 labels do not blanket hard-fix residues" in thread_spec
    assert "until manual motif authority and side-chain/contact-density evidence are materialized" not in msa_method
    assert "eco1_rt_clade9_plurality25_direct_contact5a_v1" in msa_method


def test_contact_risk_profile_is_documented_as_prior_evidence_only() -> None:
    mask_policy = _text("docs/studies/eco1_rt_repack/contexts/residue-mask-policy.md")
    status = _text("docs/studies/eco1_rt_repack/record/status.md")
    command_readme = _text("docs/studies/eco1_rt_repack/operations/runtime/command-groups/README.md")

    assert "evidence reviews only" in mask_policy
    assert "Evidence-review artifacts explain the structure context but are not mask inputs" in mask_policy
    assert "direct contact instead: only mapped residues within 5 A" in mask_policy
    assert "contact_geometry_profile.parquet" in status
    assert "do not protect or release residues" in status
    assert "current mask rule" in command_readme
