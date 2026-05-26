"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/catalog/msd_ids.py

Parser and static lint helpers for Retron MSD construct labels.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from dnadesign.contracts.sequence.msd_design_reference_v1 import compute_scar_nick_profile_s3s2s1s0

_MSD_LABEL_RE = re.compile(
    r"^(?P<construct_id>[A-Za-z0-9_.-]+)-msd\[(?P<payload_id>[A-Za-z0-9_.-]+)\];\s*"
    r"(?P<cap_id>C[A-Za-z0-9_.-]+)-L(?P<left_base>[ACGTacgt]{4})-R(?P<right_base>[ACGTacgt]{4})"
    r"(?:-(?P<profile_s3s2s1s0>[MWXmwx]{4}))?$"
)


class MsdIdError(ValueError):
    """Raised when a Retron MSD construct label is malformed or inconsistent."""


@dataclass(frozen=True)
class ParsedMsdConstructLabel:
    construct_id: str
    construct_label: str
    payload_id: str
    cap_id: str
    left_base: str
    right_base: str
    profile_s3s2s1s0: str
    s0_match_required: bool = True

    @property
    def msd_design_id(self) -> str:
        return "-".join(
            [
                "msd",
                _slug_token(self.payload_id),
                _slug_token_preserve_case(self.cap_id),
                f"L{self.left_base}",
                f"R{self.right_base}",
                self.profile_s3s2s1s0,
            ]
        )


@dataclass(frozen=True)
class MsdDesignPartInput:
    construct_id: str
    payload_id: str
    cap_id: str
    left_base: str
    right_base: str
    profile_s3s2s1s0: str | None = None


def _slug_token(value: str) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def _slug_token_preserve_case(value: str) -> str:
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9]+", "-", text)
    return text.strip("-")


def compute_scar_nick_profile(*, left_base: str, right_base: str) -> str:
    return compute_scar_nick_profile_s3s2s1s0(left_base=left_base, right_base=right_base)


def canonical_msd_construct_label(parts: MsdDesignPartInput) -> str:
    construct_id = _not_blank(parts.construct_id, label="construct_id")
    payload_id = _not_blank(parts.payload_id, label="payload_id")
    cap_id = _not_blank(parts.cap_id, label="cap_id")
    left_base = _dna4(parts.left_base, label="left_base")
    right_base = _dna4(parts.right_base, label="right_base")
    observed_profile = compute_scar_nick_profile(left_base=left_base, right_base=right_base)
    declared_profile = parts.profile_s3s2s1s0
    if declared_profile is not None and declared_profile.upper() != observed_profile:
        raise MsdIdError(
            f"MSD design parts provided profile {declared_profile.upper()} but left/right bases imply "
            f"{observed_profile}."
        )
    return f"{construct_id}-msd[{payload_id}]; {cap_id}-L{left_base}-R{right_base}-{observed_profile}"


def parse_msd_design_parts(
    parts: MsdDesignPartInput,
    *,
    allow_non_ligatable_s0: bool = False,
) -> ParsedMsdConstructLabel:
    return parse_msd_construct_label(
        canonical_msd_construct_label(parts),
        allow_non_ligatable_s0=allow_non_ligatable_s0,
    )


def parse_msd_construct_label(
    label: str,
    *,
    allow_non_ligatable_s0: bool = False,
) -> ParsedMsdConstructLabel:
    text = str(label or "").strip()
    match = _MSD_LABEL_RE.fullmatch(text)
    if match is None:
        raise MsdIdError(
            "MSD construct label must match '<construct_id>-msd[<payload>]; C<cap>-L<left4>-R<right4>[-<MWX profile>]'."
        )
    left_base = match.group("left_base").upper()
    right_base = match.group("right_base").upper()
    observed_profile = compute_scar_nick_profile(left_base=left_base, right_base=right_base)
    provided_profile = match.group("profile_s3s2s1s0")
    if provided_profile is not None and provided_profile.upper() != observed_profile:
        raise MsdIdError(
            f"MSD construct label provided profile {provided_profile.upper()} but left/right bases imply "
            f"{observed_profile}."
        )
    if observed_profile[3] != "M" and not allow_non_ligatable_s0:
        raise MsdIdError(f"MSD construct label must be scar-compatible with S0=M; observed {observed_profile}.")
    return ParsedMsdConstructLabel(
        construct_id=match.group("construct_id"),
        construct_label=text,
        payload_id=match.group("payload_id"),
        cap_id=match.group("cap_id"),
        left_base=left_base,
        right_base=right_base,
        profile_s3s2s1s0=observed_profile,
        s0_match_required=observed_profile[3] == "M",
    )


def _not_blank(value: str, *, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise MsdIdError(f"MSD design part {label} cannot be empty.")
    return text


def _dna4(value: str, *, label: str) -> str:
    text = _not_blank(value, label=label).upper()
    if not re.fullmatch(r"[ACGT]{4}", text):
        raise MsdIdError(f"MSD design part {label} must contain exactly four A/C/G/T bases.")
    return text


__all__ = [
    "MsdIdError",
    "MsdDesignPartInput",
    "ParsedMsdConstructLabel",
    "canonical_msd_construct_label",
    "compute_scar_nick_profile",
    "parse_msd_design_parts",
    "parse_msd_construct_label",
]
