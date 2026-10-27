"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/realization/placement_search.py

Template context and sequence-match helpers for construct placement contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from ..contracts.errors import ValidationError


def template_context_sequence(
    template_seq: str,
    *,
    anchor: int,
    length: int,
    circular: bool,
    direction: str,
) -> str:
    if length < 0:
        raise ValidationError(f"template context length must be >= 0, got {length}.")
    if direction not in {"upstream", "downstream"}:
        raise ValidationError("template context direction must be 'upstream' or 'downstream'.")
    if length == 0:
        return ""
    template_len = len(template_seq)
    if template_len == 0:
        raise ValidationError("template sequence cannot be empty when extracting non-empty context.")
    if not circular:
        if direction == "upstream":
            if anchor < length:
                raise ValidationError(
                    f"Requested upstream template context length {length} exceeds the available "
                    f"forward-strand prefix before placement coordinate {anchor}."
                )
            return template_seq[anchor - length : anchor]
        if anchor + length > template_len:
            raise ValidationError(
                f"Requested downstream template context length {length} exceeds the available "
                f"forward-strand suffix after placement coordinate {anchor}."
            )
        return template_seq[anchor : anchor + length]

    start = anchor - length if direction == "upstream" else anchor
    return "".join(template_seq[(start + idx) % template_len] for idx in range(length))


def template_match_offsets(
    template_seq: str,
    expected: str,
    *,
    circular: bool,
) -> list[int]:
    haystack = template_seq.upper()
    needle = expected.upper()
    if not needle:
        return []
    if circular:
        if not haystack:
            raise ValidationError("template sequence cannot be empty when searching circular template matches.")
        if len(needle) > len(haystack):
            raise ValidationError(
                f"circular template match length {len(needle)} must not exceed template length {len(haystack)}."
            )
    search_text = haystack if not circular else haystack + haystack[: len(needle) - 1]
    limit = len(haystack)
    offsets: list[int] = []
    start = 0
    while True:
        idx = search_text.find(needle, start)
        if idx < 0:
            break
        if idx < limit:
            offsets.append(idx)
        start = idx + 1
    return offsets
