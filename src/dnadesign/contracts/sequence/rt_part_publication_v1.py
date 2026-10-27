"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/contracts/sequence/rt_part_publication_v1.py

Provider-neutral opaque RT-part publication contract.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class RtPartPublicationContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _not_blank(value: str, *, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} cannot be empty.")
    return text


class RtPartPublicationProvenanceV1(RtPartPublicationContractModel):
    """Opaque provider lineage pinned by a typed source digest."""

    source_ref: str
    source_contract: str
    source_sha256: str

    @field_validator("source_ref", "source_contract")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _not_blank(value, label="RT publication provenance field")

    @field_validator("source_sha256")
    @classmethod
    def _valid_digest(cls, value: str) -> str:
        text = _not_blank(value, label="provenance.source_sha256")
        if not _SHA256_RE.fullmatch(text):
            raise ValueError("provenance.source_sha256 must be a lowercase sha256 digest.")
        return text


class RtPartV1(RtPartPublicationContractModel):
    """One provider-owned RT identity without publishing private sequence bytes."""

    part_id: str
    provider_ref: str
    cds_sha256: str
    cds_length_nt: int = Field(gt=0)
    terminal_stop_codon: Literal["included", "omitted"]
    protein_sha256: str
    protein_length_aa: int = Field(gt=0)

    @field_validator("part_id", "provider_ref")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _not_blank(value, label="RT part identity field")

    @field_validator("cds_sha256", "protein_sha256")
    @classmethod
    def _valid_digest(cls, value: str) -> str:
        text = _not_blank(value, label="RT part digest")
        if not _SHA256_RE.fullmatch(text):
            raise ValueError("RT part digests must be lowercase sha256 values.")
        return text

    @model_validator(mode="after")
    def _validate_lengths(self) -> "RtPartV1":
        terminal_codon_length = 3 if self.terminal_stop_codon == "included" else 0
        expected_cds_length = self.protein_length_aa * 3 + terminal_codon_length
        if self.cds_length_nt != expected_cds_length:
            raise ValueError(
                f"declared CDS length {self.cds_length_nt} does not match protein length "
                f"{self.protein_length_aa} under terminal_stop_codon={self.terminal_stop_codon!r} "
                f"({expected_cds_length} nt)."
            )
        return self


class RtPartPublicationV1(RtPartPublicationContractModel):
    """Stable metadata handoff for provider-owned RT parts."""

    contract: Literal["rt_part_publication_v1"] = "rt_part_publication_v1"
    schema_version: Literal[1] = 1
    owner_study_id: str
    publication_id: str
    provenance: RtPartPublicationProvenanceV1
    parts: list[RtPartV1] = Field(min_length=1)

    @field_validator("owner_study_id", "publication_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _not_blank(value, label="RT publication identity field")

    @model_validator(mode="after")
    def _validate_unique_part_identity(self) -> "RtPartPublicationV1":
        part_ids: set[str] = set()
        provider_refs: set[str] = set()
        for part in self.parts:
            if part.part_id in part_ids:
                raise ValueError(f"Duplicate part_id '{part.part_id}'.")
            if part.provider_ref in provider_refs:
                raise ValueError(f"Duplicate provider_ref '{part.provider_ref}'.")
            part_ids.add(part.part_id)
            provider_refs.add(part.provider_ref)
        return self


__all__ = [
    "RtPartPublicationProvenanceV1",
    "RtPartPublicationV1",
    "RtPartV1",
]
