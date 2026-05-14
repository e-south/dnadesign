"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/folding/src/viennarna_summary.py

Display-summary wording for ViennaRNA-native structure plots.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re

from dnadesign.contracts.visual import SequenceEvidenceMapV1


def structure_title(visual_contract: SequenceEvidenceMapV1 | None) -> str | None:
    if visual_contract is None:
        return None
    configured = visual_contract.meta.get("structure_title")
    if isinstance(configured, str) and configured.strip():
        return configured.strip()
    raw = str(visual_contract.display.title or visual_contract.state_id).strip()
    if not raw:
        return None
    return _prettify_title(raw)


def structure_subtitle_lines(
    section_annotations: list[dict[str, object]],
    visual_contract: SequenceEvidenceMapV1 | None,
) -> list[str]:
    if visual_contract is None:
        return []
    lines: list[str] = []
    payload_name = _payload_name(section_annotations)
    left_base = _stem_base_sequence(visual_contract, section_annotations, side="left")
    right_base = _stem_base_sequence(visual_contract, section_annotations, side="right")
    first_line_parts: list[str] = []
    if payload_name:
        first_line_parts.append(f"{payload_name} payload")
    base_phrase = _base_phrase(left_base=left_base, right_base=right_base)
    if base_phrase:
        first_line_parts.append(base_phrase)
    if first_line_parts:
        lines.append(" | ".join(first_line_parts))
    cap = _cap_summary(visual_contract, section_annotations)
    if cap:
        lines.append(cap)
    return lines


def _prettify_title(raw: str) -> str:
    text = raw.replace("_", " ").replace("-", " ")
    text = re.sub(r"\bcomponent\s+span\s+qa\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bcomponent\s+span\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bmanual\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return raw.strip()
    return " ".join(_pretty_token(token) for token in text.split())


def _pretty_token(token: str) -> str:
    lowered = token.lower()
    if re.fullmatch(r"x\d+", lowered):
        return lowered
    if token.isupper() and len(token) <= 6:
        return token
    if token.isdigit():
        return token
    return token[:1].upper() + token[1:].lower()


def _payload_name(section_annotations: list[dict[str, object]]) -> str:
    labels = [
        _clean_label(str(section.get("label", "")))
        for section in section_annotations
        if str(section.get("section_kind", "")) != "stem_base"
    ]
    for label in labels:
        lowered = label.lower()
        if "primary" not in lowered and "complement" not in lowered:
            continue
        candidate = re.sub(r"\b(primary|complement|reverse complement)\b", "", label, flags=re.IGNORECASE)
        candidate = _clean_label(candidate)
        if candidate:
            return candidate
    for label in labels:
        if "payload" in label.lower():
            return "Payload"
    return ""


def _clean_label(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("_", " ").strip())


def _base_phrase(*, left_base: str, right_base: str) -> str:
    if left_base and right_base:
        return f"left {left_base} / right {right_base}"
    if left_base:
        return f"left {left_base}"
    if right_base:
        return f"right {right_base}"
    return ""


def _cap_summary(
    visual_contract: SequenceEvidenceMapV1,
    section_annotations: list[dict[str, object]],
) -> str:
    section = next(
        (item for item in section_annotations if str(item.get("section_kind", "")) == "cap"),
        None,
    )
    if section is None:
        return ""
    start = int(section["start"])
    end = int(section["end"])
    sequence = visual_contract.primary_sequence[start:end]
    label = _clean_label(str(section.get("label", "Cap"))) or "Cap"
    return f"{label} {sequence} ({end - start} nt)"


def _stem_base_sequence(
    visual_contract: SequenceEvidenceMapV1,
    section_annotations: list[dict[str, object]],
    *,
    side: str,
) -> str:
    section = _stem_base_section(section_annotations, side=side)
    if section is None:
        return ""
    start = int(section["start"])
    end = int(section["end"])
    return visual_contract.primary_sequence[start:end]


def _stem_base_section(
    section_annotations: list[dict[str, object]],
    *,
    side: str,
) -> dict[str, object] | None:
    side_text = side.lower()
    for section in section_annotations:
        if str(section.get("section_kind", "")) != "stem_base":
            continue
        label = str(section.get("label", "")).lower()
        section_id = str(section.get("section_id", "")).lower()
        if side_text in label or side_text in section_id:
            return section
    return None


__all__ = ["structure_subtitle_lines", "structure_title"]
