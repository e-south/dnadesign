"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/tests/test_public_api.py

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.permuter.api import (
    NucleotideDmsRequest,
    ProteinDmsRequest,
    generate_variants,
)


def test_public_api_generates_nucleotide_dms_without_filesystem() -> None:
    result = generate_variants(
        NucleotideDmsRequest(
            ref_name="toy_dna",
            sequence="AC",
            metadata={"study": "unit"},
        )
    )

    assert result.bio_type == "dna"
    assert len(result.records) == 6
    assert {record.ref_name for record in result.records} == {"toy_dna"}
    assert all(record.metadata["study"] == "unit" for record in result.records)
    assert all(record.sequence != "AC" for record in result.records)


def test_public_api_generates_protein_dms_for_selected_positions() -> None:
    result = generate_variants(
        ProteinDmsRequest(
            ref_name="toy_protein",
            sequence="MA",
            positions=(2,),
            metadata={"caller": "study-runtime"},
        )
    )

    assert result.bio_type == "protein"
    assert len(result.records) == 19
    assert {record.metadata["caller"] for record in result.records} == {"study-runtime"}
    assert all(record.modifications == (f"aa pos=2 wt=A alt={record.sequence[1]}",) for record in result.records)


def test_public_api_fails_fast_on_invalid_dna_sequence() -> None:
    with pytest.raises(ValueError, match="DNA"):
        generate_variants(NucleotideDmsRequest(ref_name="bad", sequence="AX"))
