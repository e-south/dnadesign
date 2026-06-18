"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/eligibility/contracts.py

Candidate eligibility contracts used before OPAL selection.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

_DNA = re.compile(r"[ACGT]+")
VALID_RESTRICTION_SITE_REGIONS = frozenset({"left_flank", "core", "right_flank"})


def _require_text(value: Any, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} must be non-empty")
    return text


def _require_dna(value: Any, *, field: str, case: str = "upper") -> str:
    text = _require_text(value, field=field)
    if case == "lower":
        if text != text.lower() or not _DNA.fullmatch(text.upper()):
            raise ValueError(f"{field} must be lowercase acgt")
        return text
    out = text.upper()
    if not _DNA.fullmatch(out):
        raise ValueError(f"{field} must be uppercase ACGT")
    return out


@dataclass(frozen=True)
class RestrictionSiteSpec:
    """One disallowed restriction motif with explicitly allowed regions."""

    enzyme: str
    motif: str
    allowed_regions: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "enzyme", _require_text(self.enzyme, field="enzyme"))
        object.__setattr__(self, "motif", _require_dna(self.motif, field=f"{self.enzyme}.motif"))
        regions = tuple(str(region).strip() for region in self.allowed_regions)
        if not regions:
            raise ValueError(f"{self.enzyme}.allowed_regions must contain at least one region")
        invalid = sorted(set(regions).difference(VALID_RESTRICTION_SITE_REGIONS))
        if invalid:
            allowed = ", ".join(sorted(VALID_RESTRICTION_SITE_REGIONS))
            raise ValueError(f"{self.enzyme}.allowed_regions contains invalid region(s): {invalid}; allowed={allowed}")
        object.__setattr__(self, "allowed_regions", regions)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RestrictionSiteSpec":
        if not isinstance(value, Mapping):
            raise ValueError("restriction site spec must be a mapping")
        raw_regions = value.get("allowed_regions")
        if raw_regions is None:
            raw_regions = [span.get("region") for span in value.get("allowed_spans", []) if isinstance(span, Mapping)]
        if isinstance(raw_regions, str):
            raw_regions = [raw_regions]
        return cls(
            enzyme=str(value["enzyme"]),
            motif=str(value["motif"]),
            allowed_regions=tuple(str(region) for region in raw_regions or ()),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "enzyme": self.enzyme,
            "motif": self.motif,
            "allowed_regions": list(self.allowed_regions),
        }


@dataclass(frozen=True)
class RestrictionSiteHit:
    """One restriction-site match in an assembled candidate insert."""

    enzyme: str
    motif: str
    start_0: int
    end_0: int
    region: str
    allowed: bool

    def to_json(self) -> dict[str, Any]:
        return {
            "enzyme": self.enzyme,
            "motif": self.motif,
            "start_0": int(self.start_0),
            "end_0": int(self.end_0),
            "region": self.region,
            "allowed": bool(self.allowed),
        }


@dataclass(frozen=True)
class RestrictionSiteScanReport:
    """Restriction-site scan result for one assembled insert."""

    candidate_id: str
    final_length: int
    hits: tuple[RestrictionSiteHit, ...]

    @property
    def unexpected_hits(self) -> tuple[RestrictionSiteHit, ...]:
        return tuple(hit for hit in self.hits if not hit.allowed)

    def to_json(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "final_length": int(self.final_length),
            "hits": [hit.to_json() for hit in self.hits],
            "unexpected_hits": [hit.to_json() for hit in self.unexpected_hits],
        }


@dataclass(frozen=True)
class CandidateEligibilityRuleResult:
    """Output from one candidate eligibility rule."""

    frame: pd.DataFrame
    report: dict[str, Any]


@dataclass(frozen=True)
class CandidateEligibilityResult:
    """Output from applying all configured candidate eligibility rules."""

    frame: pd.DataFrame
    reports: tuple[dict[str, Any], ...]


def params_sha256(params: Mapping[str, Any]) -> str:
    payload = json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
