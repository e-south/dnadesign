"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/subject_bindings/test_registry.py

Executable contracts for compositional RT-lnRNA subject bindings.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.subject_bindings import (
    SubjectBindingContractError,
    SubjectBindingRegistry,
    load_registered_subject_binding_materialization,
    load_registered_subject_bindings,
    load_resolved_subject_bindings,
    load_subject_bindings,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.subject_bindings import sources as binding_sources

_ECO1_PUBLICATION_PATH = Path("docs/studies/eco1_rt_repack/record/rt-parts/eco1-g3-distal-pair-v1.yaml")


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


def _registry_payload() -> dict[str, object]:
    path = (
        _repo_root() / "docs/studies/rt_lnrna_sponging_construct_triage/workbench/provenance/subject_bindings/"
        "retron_subject_bindings_v1.yaml"
    )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write_registry(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "bindings.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _eco1_publication_payload() -> dict[str, object]:
    payload = yaml.safe_load((_repo_root() / _ECO1_PUBLICATION_PATH).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _sha256(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _override_eco1_publication(monkeypatch: pytest.MonkeyPatch, publication: dict[str, object]) -> None:
    publication_path = (_repo_root() / _ECO1_PUBLICATION_PATH).resolve()
    original_load_yaml = binding_sources.load_yaml

    def _load_yaml(path: Path) -> object:
        if Path(path).resolve() == publication_path:
            return publication
        return original_load_yaml(path)

    monkeypatch.setattr(binding_sources, "load_yaml", _load_yaml)


def _subject_by_id(payload: dict[str, object], subject_id: str) -> dict[str, object]:
    subjects = payload["subjects"]
    assert isinstance(subjects, list)
    return next(item for item in subjects if isinstance(item, dict) and item.get("subject_id") == subject_id)


def test_registered_bindings_resolve_catalog_projection_and_repacked_compositions() -> None:
    registry = load_registered_subject_bindings(repo_root=_repo_root())

    assert registry.schema_id == "rt_lnrna_subject_binding_registry_v1"
    assert registry.binding_set_id == "retron_subject_bindings_v1"
    assert len(registry.subjects) == 49
    assert {
        "rt_lnrna_pair__eco1_wt_rt__retron26_lnrna__tetO",
        "rt_lnrna_pair__eco1rt_g3_d01__retron26_lnrna__tetO",
        "rt_lnrna_pair__eco1rt_g3_d02__retron26_lnrna__tetO",
        "rt_lnrna_pair__eco1_wt_rt__retron180_lnrna__tetO",
        "rt_lnrna_pair__eco1rt_g3_d01__retron180_lnrna__tetO",
    } <= set(registry.subjects_by_id)

    d01 = registry.subjects_by_id["rt_lnrna_pair__eco1rt_g3_d01__retron26_lnrna__tetO"]
    assert d01.rt_part.owner_study_id == "eco1_rt_repack"
    assert d01.study_variant_id == "eco1rt_g3_d01__retron26"
    assert d01.rt_part.part_id == "Eco1RT-G3-D01"
    assert d01.rt_part.authority_kind == "rt_part_publication_v1"
    assert d01.rt_part.sequence_sha256 == "sha256:0b0d9de3d19f7f22b09befa93365a45e3e01ffe9ff328a867333a578f1fe1191"
    assert d01.lnrna_part.part_id == "retron26"
    assert d01.msd_structure.orientation_in_lnrna == "reverse_complement"
    assert d01.msd_structure.lnrna_span_0 == (78, 152)
    assert d01.construct_projection_status == "representable"
    assert {(alias.namespace, alias.value) for alias in d01.aliases} == {
        ("reader.design_id", "pES-retron-205-Eco1RT-G3-D01; pBbS2c-rfp"),
        ("reader.assay_subject_id", "retron-205-Eco1RT-G3-D01"),
    }

    planned = registry.subjects_by_id["rt_lnrna_pair__eco1rt_g3_d01__retron180_lnrna__tetO"]
    assert planned.aliases == ()
    assert planned.msd_structure.lnrna_span_0 == (78, 157)


def test_materialization_resolution_reports_byte_blocks_without_losing_resolvable_subjects() -> None:
    resolution = load_registered_subject_binding_materialization(repo_root=_repo_root())

    assert len(resolution.resolved_subjects) == 46
    assert len(resolution.blocked_subjects) == 3
    assert {block.part_id for block in resolution.blocked_subjects} == {"Eco1RT-G3-D01", "Eco1RT-G3-D02"}
    assert all(block.provider_ref.startswith("provider:eco1_rt_repack/") for block in resolution.blocked_subjects)


def test_materialization_resolution_accepts_one_exact_resolvable_subject() -> None:
    subject_id = "rt_lnrna_pair__eco1_wt_rt__retron26_lnrna__tetO"

    resolution = load_registered_subject_binding_materialization(
        repo_root=_repo_root(),
        subject_ids=(subject_id,),
    )

    assert [item.binding.subject_id for item in resolution.resolved_subjects] == [subject_id]
    assert resolution.blocked_subjects == ()


def test_exact_reader_alias_resolves_d01_retron26_composition() -> None:
    registry = load_registered_subject_bindings(repo_root=_repo_root())

    subject = registry.resolve_alias(
        namespace="reader.assay_subject_id",
        value="retron-205-Eco1RT-G3-D01",
    )

    assert subject.subject_id == "rt_lnrna_pair__eco1rt_g3_d01__retron26_lnrna__tetO"
    assert subject.rt_part.part_id == "Eco1RT-G3-D01"
    assert subject.lnrna_part.part_id == "retron26"
    assert subject.msd_structure.structure_subject_id == "pES-retron-26"


def test_composite_reader_design_cannot_collapse_to_bare_construct_number(tmp_path: Path) -> None:
    payload = _registry_payload()
    subject = _subject_by_id(payload, "rt_lnrna_pair__eco1rt_g3_d01__retron26_lnrna__tetO")
    subject["study_variant_id"] = "retron205"

    with pytest.raises(SubjectBindingContractError, match="cannot use bare construct-number study_variant_id"):
        load_subject_bindings(repo_root=_repo_root(), registry_path=_write_registry(tmp_path, payload))


def test_provider_compositions_do_not_reuse_hairpin_construct_numbers_as_study_identity() -> None:
    registry = load_registered_subject_bindings(repo_root=_repo_root())
    plan_path = (
        _repo_root() / "docs/studies/retron_hairpin_design/workbench/deliverables/"
        "teto_retained_span_trim_ecoli_working_v1.yaml"
    )
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    assigned_ids = plan["artifact_families"]["benchling_genbank_import"]["assigned_retron_ids"].values()
    hairpin_construct_numbers = {f"retron{value.removeprefix('pES-retron-')}" for value in assigned_ids}
    provider_composition_ids = {
        subject.study_variant_id
        for subject in registry.subjects
        if subject.rt_part.owner_study_id != "rt_lnrna_sponging_construct_triage"
    }

    assert provider_composition_ids.isdisjoint(hairpin_construct_numbers)


def test_subject_id_resolver_is_exact_and_unknown_values_fail() -> None:
    registry = load_registered_subject_bindings(repo_root=_repo_root())

    planned = registry.resolve_subject_id("rt_lnrna_pair__eco1rt_g3_d01__retron180_lnrna__tetO")
    assert planned.aliases == ()

    with pytest.raises(SubjectBindingContractError, match="unknown subject_id"):
        registry.resolve_subject_id("RT_LNRNA_PAIR__ECO1RT_G3_D01__RETRON180_LNRNA__TETO")

    with pytest.raises(SubjectBindingContractError, match="unknown alias"):
        registry.resolve_alias(
            namespace="reader.assay_subject_id",
            value="retron-205-eco1rt-g3-d01",
        )


def test_binding_loader_rejects_unknown_fields(tmp_path: Path) -> None:
    payload = _registry_payload()
    payload["subjects"][0]["unexpected"] = True

    with pytest.raises(SubjectBindingContractError, match="unknown field"):
        load_subject_bindings(repo_root=_repo_root(), registry_path=_write_registry(tmp_path, payload))


def test_binding_loader_rejects_duplicate_subjects(tmp_path: Path) -> None:
    payload = _registry_payload()
    payload["subjects"].append(copy.deepcopy(payload["subjects"][0]))

    with pytest.raises(SubjectBindingContractError, match="duplicate subject_id"):
        load_subject_bindings(repo_root=_repo_root(), registry_path=_write_registry(tmp_path, payload))


def test_binding_loader_rejects_ambiguous_reader_aliases(tmp_path: Path) -> None:
    payload = _registry_payload()
    payload["subjects"][1]["aliases"] = copy.deepcopy(payload["subjects"][0]["aliases"])

    with pytest.raises(SubjectBindingContractError, match="ambiguous alias"):
        load_subject_bindings(repo_root=_repo_root(), registry_path=_write_registry(tmp_path, payload))


def test_public_registry_construction_rejects_ambiguous_reader_aliases() -> None:
    loaded = load_registered_subject_bindings(repo_root=_repo_root())
    first, second = loaded.subjects[:2]
    forged_second = replace(second, aliases=(first.aliases[0],))

    with pytest.raises(SubjectBindingContractError, match="ambiguous alias"):
        SubjectBindingRegistry(
            schema_id=loaded.schema_id,
            study_id=loaded.study_id,
            binding_set_id=loaded.binding_set_id,
            subjects=(first, forged_second),
        )


def test_only_loader_constructed_registry_is_source_closed() -> None:
    loaded = load_registered_subject_bindings(repo_root=_repo_root())
    direct = SubjectBindingRegistry(
        schema_id=loaded.schema_id,
        study_id=loaded.study_id,
        binding_set_id=loaded.binding_set_id,
        subjects=loaded.subjects,
    )

    assert loaded.is_source_closed is True
    assert direct.is_source_closed is False


def test_contained_file_rejects_parent_traversal_and_symlink_escape(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    outside = tmp_path / "outside.yaml"
    outside.write_text("value: outside\n", encoding="utf-8")
    (bundle / "escape.yaml").symlink_to(outside)

    with pytest.raises(SubjectBindingContractError, match="parent traversal"):
        binding_sources.contained_file(bundle, "../outside.yaml", label="fixture")
    with pytest.raises(SubjectBindingContractError, match="remain inside"):
        binding_sources.contained_file(bundle, "escape.yaml", label="fixture")


def test_binding_loader_rejects_rt_digest_drift_and_blocks_projection(tmp_path: Path) -> None:
    payload = _registry_payload()
    payload["subjects"][1]["rt_part"]["sequence_sha256"] = "sha256:" + "0" * 64

    with pytest.raises(SubjectBindingContractError, match="projection blocked.*RT CDS authority digest"):
        load_subject_bindings(repo_root=_repo_root(), registry_path=_write_registry(tmp_path, payload))


def test_binding_loader_accepts_non_eco1_provider_with_different_protein_length(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _registry_payload()
    subject = _subject_by_id(registry, "rt_lnrna_pair__eco1rt_g3_d02__retron26_lnrna__tetO")
    rt_part = subject["rt_part"]
    assert isinstance(rt_part, dict)
    cds_digest = _sha256("private-short-cds")
    synthetic_path = Path("docs/studies/eco1_rt_repack/record/rt-parts/README.md")
    rt_part.update(
        {
            "owner_study_id": "literature_rt_parts",
            "part_id": "LiteratureRT-Short",
            "authority_kind": "rt_part_publication_v1",
            "source_path": synthetic_path.as_posix(),
            "record_id": "LiteratureRT-Short",
            "sequence_sha256": cds_digest,
        }
    )
    publication = {
        "contract": "rt_part_publication_v1",
        "schema_version": 1,
        "owner_study_id": "literature_rt_parts",
        "publication_id": "literature_rt_parts_v1",
        "provenance": {
            "source_ref": "doi:example/rt-short",
            "source_contract": "curated_rt_source_v1",
            "source_sha256": _sha256("curated-source"),
        },
        "parts": [
            {
                "part_id": "LiteratureRT-Short",
                "provider_ref": "provider:literature_rt_parts/rt-short",
                "cds_sha256": cds_digest,
                "cds_length_nt": 9,
                "terminal_stop_codon": "included",
                "protein_sha256": _sha256("MK"),
                "protein_length_aa": 2,
            }
        ],
    }
    publication_path = (_repo_root() / synthetic_path).resolve()
    original_load_yaml = binding_sources.load_yaml

    def _load_yaml(path: Path) -> object:
        if Path(path).resolve() == publication_path:
            return publication
        return original_load_yaml(path)

    monkeypatch.setattr(binding_sources, "load_yaml", _load_yaml)

    observed_registry = load_subject_bindings(
        repo_root=_repo_root(),
        registry_path=_write_registry(tmp_path, registry),
    )

    observed = observed_registry.subjects_by_id[str(subject["subject_id"])]
    assert observed.rt_part.owner_study_id == "literature_rt_parts"
    assert observed.rt_part.sequence_sha256 == cds_digest


def test_binding_loader_rejects_provider_length_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _registry_payload()
    publication = _eco1_publication_payload()
    parts = publication["parts"]
    assert isinstance(parts, list)
    d02 = next(item for item in parts if isinstance(item, dict) and item.get("part_id") == "Eco1RT-G3-D02")
    d02["cds_length_nt"] = 960
    _override_eco1_publication(monkeypatch, publication)

    with pytest.raises(
        SubjectBindingContractError,
        match="declared CDS length 960 does not match protein length 320 under terminal_stop_codon='included'",
    ):
        load_subject_bindings(repo_root=_repo_root(), registry_path=_write_registry(tmp_path, registry))


def test_binding_loader_rejects_provider_publication_with_private_sequence_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _registry_payload()
    publication = _eco1_publication_payload()
    parts = publication["parts"]
    assert isinstance(parts, list)
    d02 = next(item for item in parts if isinstance(item, dict) and item.get("part_id") == "Eco1RT-G3-D02")
    d02["cds_sequence_5to3"] = "ATGAAATAA"
    _override_eco1_publication(monkeypatch, publication)

    with pytest.raises(
        SubjectBindingContractError,
        match=r"(?s)cds_sequence_5to3.*Extra inputs are not permitted",
    ):
        load_subject_bindings(repo_root=_repo_root(), registry_path=_write_registry(tmp_path, registry))


def test_resolved_binding_loader_fails_closed_when_provider_does_not_publish_rt_bytes(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        SubjectBindingContractError,
        match="RT CDS bytes are not published.*provider:eco1_rt_repack/rt-parts/Eco1RT-G3-D01",
    ):
        load_resolved_subject_bindings(
            repo_root=_repo_root(),
            registry_path=_write_registry(tmp_path, _registry_payload()),
        )


def test_binding_loader_rejects_msd_to_lnrna_orientation_or_span_drift(tmp_path: Path) -> None:
    payload = _registry_payload()
    payload["subjects"][0]["msd_structure"]["orientation_in_lnrna"] = "forward"

    with pytest.raises(SubjectBindingContractError, match="MSD sequence does not match lnRNA span"):
        load_subject_bindings(repo_root=_repo_root(), registry_path=_write_registry(tmp_path, payload))


def test_binding_loader_binds_structure_materialization_id_to_source_bundle(tmp_path: Path) -> None:
    payload = _registry_payload()
    payload["subjects"][0]["msd_structure"]["structure_materialization_id"] = "wrong-panel"

    with pytest.raises(SubjectBindingContractError, match="does not match the source bundle"):
        load_subject_bindings(repo_root=_repo_root(), registry_path=_write_registry(tmp_path, payload))


def test_binding_loader_rejects_owner_authority_mismatch(tmp_path: Path) -> None:
    payload = _registry_payload()
    payload["subjects"][1]["rt_part"]["owner_study_id"] = "rt_lnrna_sponging_construct_triage"

    with pytest.raises(SubjectBindingContractError, match="publication owner.*does not match owner_study_id"):
        load_subject_bindings(repo_root=_repo_root(), registry_path=_write_registry(tmp_path, payload))


def test_binding_loader_rejects_catalog_rt_part_identity_drift(tmp_path: Path) -> None:
    payload = _registry_payload()
    payload["subjects"][0]["rt_part"]["part_id"] = "genbank:wrong#RT"

    with pytest.raises(SubjectBindingContractError, match="RT part_id does not match"):
        load_subject_bindings(repo_root=_repo_root(), registry_path=_write_registry(tmp_path, payload))


def test_binding_loader_rejects_genbank_source_file_digest_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_ref = "docs/studies/rt_lnrna_sponging_construct_triage/workbench/provenance/genbank/pes-retron-18.gb"
    changed_source = tmp_path / "pes-retron-18.gb"
    changed_source.write_bytes((_repo_root() / source_ref).read_bytes() + b"\n")
    original_contained_file = binding_sources.contained_file

    def _contained_file(base: Path, value: str, *, label: str) -> Path:
        if value == source_ref:
            return changed_source
        return original_contained_file(base, value, label=label)

    monkeypatch.setattr(binding_sources, "contained_file", _contained_file)

    with pytest.raises(
        SubjectBindingContractError,
        match="retron18: GenBank source file digest mismatch",
    ):
        load_registered_subject_bindings(repo_root=_repo_root())
