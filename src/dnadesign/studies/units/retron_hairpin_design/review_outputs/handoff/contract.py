"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/review_outputs/handoff/contract.py

Sequence-handoff contract constants for Retron review outputs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

SEQUENCE_HANDOFF_COLUMNS = (
    "order",
    "variant_id",
    "construct_id",
    "msd_design_id",
    "scaffold",
    "retained_window",
    "insert_nt",
    "role",
    "genbank",
    "reverse_complement_genbank",
    "forward_fasta",
    "reverse_complement_fasta",
    "features_csv",
)

SEQUENCE_HANDOFF_REQUIRED_FIELDS = (
    "genbank",
    "reverse_complement_genbank",
    "forward_fasta",
    "reverse_complement_fasta",
    "features_csv",
)

SEQUENCE_HANDOFF_MANIFEST_KEY = "sequence_handoff"

__all__ = [
    "SEQUENCE_HANDOFF_COLUMNS",
    "SEQUENCE_HANDOFF_MANIFEST_KEY",
    "SEQUENCE_HANDOFF_REQUIRED_FIELDS",
]
