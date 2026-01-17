"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/adapters/outputs/record.py

Canonical output record for DenseGen outputs.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from ...core.canonical import compute_id, normalize_sequence


def _namespace_meta(meta: Dict[str, Any], namespace: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in meta.items():
        nk = f"{namespace}__{k}" if not k.startswith(f"{namespace}__") else k
        out[nk] = v
    return out


@dataclass(frozen=True)
class OutputRecord:
    id: str
    sequence: str
    bio_type: str
    alphabet: str
    source: str
    meta: Dict[str, Any]

    @classmethod
    def from_sequence(
        cls,
        *,
        sequence: str,
        meta: Dict[str, Any],
        source: str,
        bio_type: str,
        alphabet: str,
    ) -> "OutputRecord":
        seq_norm = normalize_sequence(sequence, bio_type, alphabet)
        rid = compute_id(bio_type, seq_norm)
        return cls(
            id=rid,
            sequence=seq_norm,
            bio_type=bio_type,
            alphabet=alphabet,
            source=source,
            meta=dict(meta),
        )

    def to_row(self, namespace: str) -> Dict[str, Any]:
        row = {
            "id": self.id,
            "sequence": self.sequence,
            "bio_type": self.bio_type,
            "alphabet": self.alphabet,
            "source": self.source,
        }
        row.update(_namespace_meta(self.meta, namespace))
        return row
