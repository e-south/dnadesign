"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/api/contracts.py

Public API request/result contracts.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping

Metadata = Mapping[str, object]


@dataclass(frozen=True)
class NucleotideDmsRequest:
    ref_name: str
    sequence: str
    regions: tuple[tuple[int, int], ...] = ()
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True)
class ProteinDmsRequest:
    ref_name: str
    sequence: str
    positions: tuple[int, ...] = ()
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True)
class VariantRecord:
    id: str
    ref_name: str
    bio_type: Literal["dna", "protein"]
    sequence: str
    modifications: tuple[str, ...]
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True)
class PermuterResult:
    request_id: str
    ref_name: str
    bio_type: Literal["dna", "protein"]
    reference_sequence: str
    records: tuple[VariantRecord, ...]
