"""Human-readable labels for DenseGen TFBS probe surfaces."""

from __future__ import annotations

import re

_REGULATOR_LABELS = {
    "lexA": "LexA",
    "cpxR": "CpxR",
    "baeR": "BaeR",
}

_COMPOSITE_REGULATOR_LABELS = {
    "cpxR_or_baeR": "CpxR or BaeR",
}

_CONTROL_LABELS = {
    "matched_label_permutation_negative_control": "scrambled control",
    "count_preserving_slot_confound_control": "count-preserving slot diagnostic control",
    "count_fixed_shuffled_slot_negative_control": "count-fixed shuffled-slot control",
}


def tfbs_label_display(label_name: object) -> str:
    """Return a human-readable TFBS label while preserving literal semantics."""

    text = str(label_name or "").strip()
    if not text:
        return "TFBS label"
    if text.endswith("_count_fraction"):
        return f"{_display_regulator(text.removesuffix('_count_fraction'))} count fraction"
    if text.endswith("_count"):
        return f"{_display_regulator(text.removesuffix('_count'))} count"
    if text.endswith("_present"):
        return f"{_display_regulator(text.removesuffix('_present'))} presence"
    slot_match = re.fullmatch(r"(.+)_in_slot([0-9]+)", text)
    if slot_match:
        return f"{_display_regulator(slot_match.group(1))} in slot {int(slot_match.group(2))}"
    return _fallback_display(text)


def tfbs_label_expression(label_name: object) -> str | None:
    """Return a compact mathematical expression for labels with explicit algebra."""

    text = str(label_name or "").strip()
    if not text.endswith("_count_fraction"):
        return None
    regulator = text.removesuffix("_count_fraction")
    if regulator == "cpxR_or_baeR":
        return "(CpxR + BaeR) count / 3"
    return f"{_display_regulator(regulator)} count / 3"


def tfbs_label_title(label_name: object) -> str:
    """Return a title-ready label with inline algebra when available."""

    display = tfbs_label_display(label_name)
    expression = tfbs_label_expression(label_name)
    return f"{display} ({expression})" if expression else display


def tfbs_label_compact_title(label_name: object) -> str:
    """Return a compact manuscript label without dropping literal label semantics."""

    return tfbs_label_display(label_name).replace(" count fraction", " count-fraction")


def tfbs_label_dropdown_title(label_name: object) -> str:
    """Return a compact dropdown label while preserving count-fraction algebra."""

    label_text = str(label_name or "").strip()
    display = tfbs_label_display(label_name)
    expression = tfbs_label_expression(label_name)
    if expression is None:
        return display
    if label_text.removesuffix("_count_fraction") == "cpxR_or_baeR":
        return f"{display} (combined count / 3)"
    compact_expression = expression
    for regulator in (*_REGULATOR_LABELS.values(), "CpxR + BaeR"):
        compact_expression = compact_expression.replace(f"{regulator} count", "count")
    compact_expression = compact_expression.replace("(count) /", "count /")
    return f"{display} ({compact_expression})"


def tfbs_control_display_label(control_role: object, *, label_name: object | None = None) -> str:
    """Return a reader-facing label for the active control surface."""

    role = str(control_role or "").strip()
    if role in _CONTROL_LABELS:
        return _CONTROL_LABELS[role]
    if "_in_slot" in str(label_name or ""):
        return "slot-position control"
    if role:
        return _fallback_display(role)
    return "control"


def tfbs_control_pair_label(control_role: object, *, label_name: object | None = None) -> str:
    """Return a compact DenseGen-vs-control comparison label."""

    return f"DenseGen label vs {tfbs_control_display_label(control_role, label_name=label_name)}"


def _display_regulator(value: str) -> str:
    token = str(value).strip()
    if token in _REGULATOR_LABELS:
        return _REGULATOR_LABELS[token]
    if token in _COMPOSITE_REGULATOR_LABELS:
        return _COMPOSITE_REGULATOR_LABELS[token]
    return _fallback_display(token)


def _fallback_display(value: str) -> str:
    words = str(value).replace("_", " ").split()
    return " ".join(_REGULATOR_LABELS.get(word, word[:1].upper() + word[1:]) for word in words) or "TFBS label"
