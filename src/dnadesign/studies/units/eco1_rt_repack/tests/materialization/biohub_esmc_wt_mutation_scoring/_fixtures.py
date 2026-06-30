"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/biohub_esmc_wt_mutation_scoring/_fixtures.py

Eco1 Biohub ESMC WT mutation-scoring test fixtures.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from dnadesign.permuter import CANONICAL_AMINO_ACIDS
from dnadesign.thread.adapters.biohub_esmc import BiohubCredential


class FakeSequenceLogitsClient:
    def __init__(self) -> None:
        self.credential = BiohubCredential(key_label="bu-dunlop-lab", token="fixture-secret")
        self.requested_sequences: list[str] = []

    def amino_acid_token_indices(self, *, model: str) -> dict[str, int]:
        del model
        return {aa: index for index, aa in enumerate(CANONICAL_AMINO_ACIDS)}

    def sequence_logits_for_sequence(
        self,
        sequence: str,
        *,
        model: str,
    ) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
        del model
        normalized = sequence.strip().upper()
        self.requested_sequences.append(normalized)
        tokens = [0, *range(1, len(normalized) + 1), 2]
        logits = [_flat_logits() for _token in tokens]
        masked_index = normalized.index("_") + 1
        logits[masked_index] = _favored_logits("W")
        return (
            {"outputs": {"sequence": tokens}, "potential_sequence_of_concern": False},
            {"logits": {"sequence": logits}, "embeddings": None, "hidden_states": None},
            tokens,
        )


class TimeoutOnceSequenceLogitsClient(FakeSequenceLogitsClient):
    def sequence_logits_for_sequence(
        self,
        sequence: str,
        *,
        model: str,
    ) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
        if self.requested_sequences:
            raise OSError("read operation timed out")
        return super().sequence_logits_for_sequence(sequence, model=model)


def write_mask_set(path: Path, *, length: int) -> None:
    residues = [
        {
            "canonical_position": position,
            "wt_aa": "A",
            "protected": position == 1,
            "non_fixed": position != 1,
            "non_fixed_missing_backbone": False,
            "protection_reasons": ["fixture"] if position == 1 else [],
            "motif_protected": position == 1,
            "wang_ec86_direct_contact_prior": False,
            "direct_retained_dna_rna_contact_5a": False,
            "evolutionarily_conserved_clade9_25pct_plurality": False,
            "wt_plurality_frequency": 0.25,
            "min_distance_to_retained_dna_rna_angstrom": 6.0,
            "rt_interval_review_label": "RT1" if position in {2, 3} else "",
        }
        for position in range(1, length + 1)
    ]
    path.write_text(
        yaml.safe_dump({"schema_id": "thread.mask_set", "residues": residues}, sort_keys=False),
        encoding="utf-8",
    )


def rewrite_position_table_with_old_fraction_name(path: Path) -> None:
    table = pq.read_table(path)
    metadata = table.schema.metadata
    values = table.column("fraction_negative_alternate_llr")
    old_table = table.drop(["fraction_negative_alternate_llr"]).append_column("fraction_negative_llr", values)
    pq.write_table(old_table.replace_schema_metadata(metadata), path)


def rewrite_position_table_with_null_alternate_fraction(path: Path) -> None:
    rows = pq.read_table(path).to_pylist()
    for row in rows:
        row["fraction_negative_alternate_llr"] = None
    metadata = pq.read_schema(path).metadata
    table = pa.Table.from_pylist(rows)
    pq.write_table(table.replace_schema_metadata(metadata), path)


def _flat_logits() -> list[float]:
    return [0.0 for _aa in CANONICAL_AMINO_ACIDS]


def _favored_logits(aa: str) -> list[float]:
    values = _flat_logits()
    values[list(CANONICAL_AMINO_ACIDS).index(aa)] = 3.0
    return values
