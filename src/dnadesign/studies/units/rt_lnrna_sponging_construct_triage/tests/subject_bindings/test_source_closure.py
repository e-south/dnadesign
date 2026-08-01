"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/subject_bindings/test_source_closure.py

Fail-fast source closure for projected RT-lnRNA subject bindings.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.subject_bindings import (
    load_resolved_registered_subject_bindings,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.subject_bindings import sources as binding_sources
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.subject_bindings.contracts import (
    SubjectBindingContractError,
)


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


def test_local_hairpin_handoff_digest_drift_blocks_binding_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record_path = (
        _repo_root() / "docs/studies/retron_hairpin_design/workbench/provenance/msd_region_records/"
        "retron_msd_structure_panel_v1/variants/pes-retron-195-msd-region.yaml"
    ).resolve()
    original_load_yaml = binding_sources.load_yaml

    def _load_yaml(path: Path) -> object:
        payload = original_load_yaml(path)
        if Path(path).resolve() != record_path:
            return payload
        drifted = copy.deepcopy(payload)
        assert isinstance(drifted, dict)
        drifted["source_sequence_sha256"] = "0" * 64
        return drifted

    monkeypatch.setattr(binding_sources, "load_yaml", _load_yaml)

    with pytest.raises(
        SubjectBindingContractError,
        match="hairpin source sequence digest disagrees with catalog lnRNA digest",
    ):
        load_resolved_registered_subject_bindings(repo_root=_repo_root())


def test_catalog_handoff_marker_cannot_disable_hairpin_source_closure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog_path = (
        _repo_root() / "docs/studies/rt_lnrna_sponging_construct_triage/workbench/provenance/"
        "retron-variant-genbank-catalog.yaml"
    ).resolve()
    record_path = (
        _repo_root() / "docs/studies/retron_hairpin_design/workbench/provenance/msd_region_records/"
        "retron_msd_structure_panel_v1/variants/pes-retron-195-msd-region.yaml"
    ).resolve()
    original_load_yaml = binding_sources.load_yaml

    def _load_yaml(path: Path) -> object:
        payload = original_load_yaml(path)
        resolved = Path(path).resolve()
        if resolved == catalog_path:
            drifted = copy.deepcopy(payload)
            assert isinstance(drifted, dict)
            drifted["records"]["retron195"]["benchling_url"] = "local_genbank_only"
            return drifted
        if resolved == record_path:
            drifted = copy.deepcopy(payload)
            assert isinstance(drifted, dict)
            drifted["source_sequence_sha256"] = "0" * 64
            return drifted
        return payload

    monkeypatch.setattr(binding_sources, "load_yaml", _load_yaml)

    with pytest.raises(
        SubjectBindingContractError,
        match="projected identity digest drifted",
    ):
        load_resolved_registered_subject_bindings(repo_root=_repo_root())


def test_catalog_projection_identity_drift_blocks_binding_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_path = (
        _repo_root() / "docs/studies/rt_lnrna_sponging_construct_triage/workbench/provenance/subject_bindings/"
        "retron_subject_bindings_v1.yaml"
    ).resolve()
    original_load_yaml = binding_sources.load_yaml

    def _load_yaml(path: Path) -> object:
        payload = original_load_yaml(path)
        if Path(path).resolve() != registry_path:
            return payload
        drifted = copy.deepcopy(payload)
        assert isinstance(drifted, dict)
        drifted["source_sets"][0]["projection_sha256"] = "sha256:" + "0" * 64
        return drifted

    monkeypatch.setattr(binding_sources, "load_yaml", _load_yaml)

    with pytest.raises(
        SubjectBindingContractError,
        match="projected identity digest drifted",
    ):
        load_resolved_registered_subject_bindings(repo_root=_repo_root())
