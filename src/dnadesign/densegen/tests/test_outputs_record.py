from __future__ import annotations

from dnadesign.densegen.src.adapters.outputs import OutputRecord


def test_output_record_namespaces_meta() -> None:
    rec = OutputRecord.from_sequence(
        sequence="ATGC",
        meta={"foo": 1, "densegen__bar": 2},
        source="src",
        bio_type="dna",
        alphabet="dna_4",
    )
    row = rec.to_row("densegen")
    assert row["densegen__foo"] == 1
    assert row["densegen__bar"] == 2
