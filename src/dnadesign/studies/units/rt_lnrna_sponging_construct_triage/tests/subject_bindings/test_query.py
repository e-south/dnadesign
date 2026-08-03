"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/subject_bindings/test_query.py

Exact query tests for compositional RT-lnRNA subjects.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.subject_bindings import SubjectBindingContractError
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.subject_bindings.query import (
    query_registered_subjects,
)


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


def test_query_subjects_by_exact_component_and_alias() -> None:
    repo_root = _repo_root()
    d01 = query_registered_subjects(repo_root=repo_root, rt_part_id="Eco1RT-G3-D01")
    assert d01["match_count"] == 2
    assert {row["lnrna_part"]["part_id"] for row in d01["subjects"]} == {"retron26", "retron180"}

    retron26 = query_registered_subjects(repo_root=repo_root, lnrna_part_id="retron26")
    assert retron26["match_count"] == 3

    reader_alias = query_registered_subjects(
        repo_root=repo_root,
        reader_assay_subject_id="retron-205-Eco1RT-G3-D01",
    )
    assert reader_alias["match_count"] == 1
    assert reader_alias["subjects"][0]["subject_id"] == ("rt_lnrna_pair__eco1rt_g3_d01__retron26_lnrna__tetO")


def test_query_subjects_rejects_ambiguous_or_inexact_selectors() -> None:
    repo_root = _repo_root()
    with pytest.raises(SubjectBindingContractError, match="exactly one selector"):
        query_registered_subjects(repo_root=repo_root, rt_part_id="Eco1RT-G3-D01", lnrna_part_id="retron26")
    with pytest.raises(SubjectBindingContractError, match="no subject matches exact"):
        query_registered_subjects(repo_root=repo_root, lnrna_part_id="Retron26")
    with pytest.raises(SubjectBindingContractError, match="without outer whitespace"):
        query_registered_subjects(repo_root=repo_root, lnrna_part_id=" retron26")
