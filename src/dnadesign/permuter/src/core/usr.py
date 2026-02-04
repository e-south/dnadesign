"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/core/usr.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone


def _alphabet_for(bio_type: str) -> str:
    bt = (bio_type or "").lower()
    if bt == "dna":
        return "dna_4"
    if bt == "protein":
        return "protein_20"
    return "unknown"


def _sha1_id(bio_type: str, sequence: str) -> str:
    return hashlib.sha1(f"{bio_type}|{sequence}".encode("utf-8")).hexdigest()


def make_usr_row(*, sequence: str, bio_type: str, source: str) -> dict:
    sequence = str(sequence)
    bio_type = ("dna" if bio_type.lower() == "dna" else "protein") if bio_type else "dna"
    return {
        "id": _sha1_id(bio_type, sequence),
        "bio_type": bio_type,
        "sequence": sequence,
        "alphabet": _alphabet_for(bio_type),
        "length": len(sequence),
        "source": source,
        "created_at": datetime.now(timezone.utc),
    }
