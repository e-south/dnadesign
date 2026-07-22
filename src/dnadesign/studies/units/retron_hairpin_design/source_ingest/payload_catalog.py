"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/source_ingest/payload_catalog.py

Load study-owned payload binding-site catalogs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import yaml

from .models import MsdRegionIngestError
from .payload_binding_models import MotifModel, PayloadBindingCatalog, PayloadMember
from .payload_binding_utils import (
    normalize_dna,
    optional_text,
    require_mapping,
    require_sequence,
    resolve_catalog_path,
    reverse_complement,
    span_dict,
)


def load_payload_binding_catalog(path: str | Path) -> PayloadBindingCatalog:
    """Load the study-owned payload binding-site catalog."""

    catalog_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(catalog_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise MsdRegionIngestError(f"Payload binding catalog is not a YAML mapping: {catalog_path}")
    if payload.get("contract") != "retron_payload_binding_site_catalog_v1":
        raise MsdRegionIngestError(f"{catalog_path} must declare contract=retron_payload_binding_site_catalog_v1.")
    motif_models = _load_motif_models(payload.get("motif_models") or {}, catalog_path=catalog_path)
    default_motif_model_id = optional_text(payload.get("default_motif_model_id"))
    if default_motif_model_id is not None and default_motif_model_id not in motif_models:
        raise MsdRegionIngestError(f"Unknown default_motif_model_id in {catalog_path}: {default_motif_model_id}")
    members = _load_payload_members(payload.get("payload_families") or {}, motif_models=motif_models)
    reference_payload_ids = tuple(
        str(item.get("reference_payload_id") or "").strip()
        for item in require_sequence(payload.get("reference_payloads") or (), "reference_payloads")
    )
    unknown_references = sorted(payload_id for payload_id in reference_payload_ids if payload_id not in members)
    if unknown_references:
        raise MsdRegionIngestError(
            f"{catalog_path} reference_payloads are absent from payload_families: {', '.join(unknown_references)}"
        )
    return PayloadBindingCatalog(
        motif_models=motif_models,
        members_by_id=members,
        members_by_primary_sequence={member.primary_sequence_5to3: member for member in members.values()},
        reference_payload_ids=reference_payload_ids,
        default_motif_model_id=default_motif_model_id,
    )


def _load_motif_models(raw: object, *, catalog_path: Path) -> dict[str, MotifModel]:
    models: dict[str, MotifModel] = {}
    for motif_model_id, item in require_mapping(raw, "motif_models").items():
        if not isinstance(item, Mapping):
            raise MsdRegionIngestError(f"motif_models.{motif_model_id} must be a mapping.")
        source_ref = str(item.get("source_ref") or motif_model_id).strip()
        meme_path_text = str(item.get("meme_path") or "").strip()
        if not meme_path_text:
            raise MsdRegionIngestError(f"motif_models.{motif_model_id}.meme_path is required.")
        meme_path = resolve_catalog_path(meme_path_text, catalog_path=catalog_path)
        models[str(motif_model_id)] = MotifModel(
            motif_model_id=str(motif_model_id),
            source_ref=source_ref,
            matrix=_load_meme_probability_matrix(meme_path),
            congruence_threshold_fraction=float(item.get("congruence_threshold_fraction", 0.65)),
        )
    return models


def _load_payload_members(raw: object, *, motif_models: Mapping[str, MotifModel]) -> dict[str, PayloadMember]:
    members: dict[str, PayloadMember] = {}
    for family_id, family in require_mapping(raw, "payload_families").items():
        if not isinstance(family, Mapping):
            raise MsdRegionIngestError(f"payload_families.{family_id} must be a mapping.")
        parent_payload_id = str(family.get("parent_payload_id") or "").strip()
        parent_sequence = normalize_dna(str(family.get("primary_sequence_5to3") or ""))
        if not parent_payload_id or not parent_sequence:
            raise MsdRegionIngestError(
                f"payload_families.{family_id} needs parent_payload_id and primary_sequence_5to3."
            )
        motif_model_id = optional_text(family.get("motif_model_id"))
        if motif_model_id is not None and motif_model_id not in motif_models:
            raise MsdRegionIngestError(f"payload_families.{family_id} unknown motif_model_id: {motif_model_id}")
        _load_family_members(
            members,
            family_id=str(family_id),
            family=family,
            parent_payload_id=parent_payload_id,
            parent_sequence=parent_sequence,
            motif_model_id=motif_model_id,
        )
    return members


def _load_family_members(
    members: dict[str, PayloadMember],
    *,
    family_id: str,
    family: Mapping[str, object],
    parent_payload_id: str,
    parent_sequence: str,
    motif_model_id: str | None,
) -> None:
    member_rows = require_mapping(family.get("members") or {}, f"payload_families.{family_id}.members")
    for member_id, member_row in member_rows.items():
        if not isinstance(member_row, Mapping):
            raise MsdRegionIngestError(f"payload_families.{family_id}.members.{member_id} must be a mapping.")
        span = span_dict(member_row.get("retained_parent_span_0"), default_end=len(parent_sequence))
        sequence = normalize_dna(
            str(member_row.get("exact_sequence_5to3") or parent_sequence[span["start"] : span["end"]])
        )
        member = PayloadMember(
            family_id=family_id,
            parent_payload_id=parent_payload_id,
            member_id=str(member_id),
            primary_sequence_5to3=sequence,
            complement_sequence_5to3=normalize_dna(
                str(member_row.get("complement_sequence_5to3") or reverse_complement(sequence))
            ),
            retained_parent_span_0=span,
            motif_model_id=optional_text(member_row.get("motif_model_id")) or motif_model_id,
            parent_primary_sequence_5to3=parent_sequence,
        )
        if member.member_id in members:
            raise MsdRegionIngestError(f"Duplicate payload member id in payload catalog: {member.member_id}")
        members[member.member_id] = member


def _load_meme_probability_matrix(path: Path) -> tuple[tuple[float, float, float, float], ...]:
    rows: list[tuple[float, float, float, float]] = []
    in_matrix = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("letter-probability matrix:"):
            in_matrix = True
            continue
        if not in_matrix:
            continue
        stripped = line.strip()
        if not stripped:
            if rows:
                break
            continue
        values = tuple(float(value) for value in stripped.split())
        if len(values) != 4:
            break
        rows.append(values)
    if not rows:
        raise MsdRegionIngestError(f"No MEME probability matrix found: {path}")
    return tuple(rows)


__all__ = ["load_payload_binding_catalog"]
