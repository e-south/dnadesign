"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/mismatch_notation.py

Compact operator-facing mismatch notation helpers for YIU.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeAlias

from dnadesign.cruncher.yiu.domain_models import MismatchSelection

MismatchLike: TypeAlias = MismatchSelection | Mapping[str, object]

_STRAND_LABELS = {
    "payload": "PS",
    "complement": "AS",
}


def _payload(entry: MismatchLike) -> Mapping[str, object]:
    if isinstance(entry, Mapping):
        return entry
    return entry.model_dump(mode="json")


def _strand_label(mutated_strand: str) -> str:
    try:
        return _STRAND_LABELS[mutated_strand]
    except KeyError as exc:
        raise ValueError(f"unsupported YIU mismatch strand for handoff notation: {mutated_strand!r}") from exc


def compact_mismatch_notation_groups(mismatches: Sequence[MismatchLike]) -> list[str]:
    grouped: dict[str, list[str]] = {}
    ordered = sorted(mismatches, key=lambda entry: int(_payload(entry)["payload_index"]))
    for entry in ordered:
        payload = _payload(entry)
        strand_label = _strand_label(str(payload["mutated_strand"]))
        edit = f"{int(payload['payload_index']) + 1}{payload['native_base']}>{payload['mutated_base']}"
        grouped.setdefault(strand_label, []).append(edit)
    return [f"{strand_label}{','.join(edits)}" for strand_label, edits in grouped.items()]


def compact_mismatch_notation_text(mismatches: Sequence[MismatchLike]) -> str:
    return "; ".join(compact_mismatch_notation_groups(mismatches))


__all__ = [
    "compact_mismatch_notation_groups",
    "compact_mismatch_notation_text",
]
