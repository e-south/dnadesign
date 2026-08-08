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
    first_line_parts = _display_fact_phrases(visual_contract)
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


def _clean_label(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("_", " ").strip())


def _display_fact_phrases(visual_contract: SequenceEvidenceMapV1) -> list[str]:
    facts = visual_contract.meta.get("facts")
    if not isinstance(facts, list):
        return []
    return [
        f"{_clean_label(str(fact.get('label') or ''))} {_clean_label(str(fact.get('value') or ''))}".strip()
        for fact in facts
        if isinstance(fact, dict)
        and _clean_label(str(fact.get("label") or ""))
        and _clean_label(str(fact.get("value") or ""))
    ]


def _cap_summary(
    visual_contract: SequenceEvidenceMapV1,
    section_annotations: list[dict[str, object]],
) -> str:
    section = next((item for item in section_annotations if str(item.get("section_kind", "")) == "cap"), None)
    if section is None:
        section = next(
            (item for item in section_annotations if str(item.get("section_kind", "")) == "cap_foldback"),
            None,
        )
    if section is None:
        return ""
    start = int(section["start"])
    end = int(section["end"])
    sequence = visual_contract.primary_sequence[start:end]
    label = _clean_label(str(section.get("label", "Cap"))) or "Cap"
    return f"{label} {sequence} ({end - start} nt)"


__all__ = ["structure_subtitle_lines", "structure_title"]
