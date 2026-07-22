"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/tests/runtime/test_orientation.py

Sequence orientation contract tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from Bio import SeqIO

from dnadesign.construct.src.composition.runtime import run_linear_ssdna_composition
from dnadesign.construct.src.sequences import orientation


@pytest.mark.parametrize(
    ("sequence", "expected"),
    [
        ("ACGTRYSWKMBDHVN", "TGCAYRSWMKVHDBN"),
        ("acgtryswkmbdhvn", "tgcayrswmkvhdbn"),
        ("aCgTrYsWkMbDhVn", "tGcAyRsWmKvHdBn"),
    ],
)
def test_complement_maps_complete_iupac_dna_and_preserves_case(sequence: str, expected: str) -> None:
    assert orientation.complement(sequence) == expected
    assert orientation.complement(expected) == sequence


@pytest.mark.parametrize(
    ("sequence", "expected"),
    [
        ("ACGTRYSWKMBDHVN", "NBDHVKMWSRYACGT"),
        ("acgtryswkmbdhvn", "nbdhvkmwsryacgt"),
        ("aCgTrYsWkMbDhVn", "nBdHvKmWsRyAcGt"),
    ],
)
def test_reverse_complement_maps_complete_iupac_dna_and_preserves_case(sequence: str, expected: str) -> None:
    assert orientation.reverse_complement(sequence) == expected
    assert orientation.reverse_complement(expected) == sequence


def test_composition_preserves_iupac_reverse_complements_across_artifacts(tmp_path: Path) -> None:
    source = "AaRrYySsWwKkMmBbDdHhVvNn"  # pragma: allowlist secret
    source_reverse_complement = "nNbBdDhHvVkKmMwWsSrRyYtT"  # pragma: allowlist secret
    assembled = source + source_reverse_complement + "Gt"
    assembled_complement = "TtYyRrSsWwMmKkVvHhDdBbNnnNvVhHdDbBmMkKwWsSyYrRaACa"  # pragma: allowlist secret
    assembled_reverse_complement = "aCAaRrYySsWwKkMmBbDdHhVvNnnNbBdDhHvVkKmMwWsSrRyYtT"  # pragma: allowlist secret
    config_path = tmp_path / "iupac_reverse_complement.yaml"
    config_path.write_text(
        f"""
contract: linear_ssdna_composition_v1
schema_version: 1
composition_id: iupac_reverse_complement
canonicalization:
  compare_sequences_case_insensitive: true
  output_sequence_preserves_case: true
units:
  - unit_id: iupac_unit
    segments:
      - segment_id: source
        sequence: {source}
      - segment_id: derived
        transform:
          kind: reverse_complement
          source_segment_id: source
          assert_expected_sequence: true
      - segment_id: tail
        sequence: Gt
    assertions:
      - assertion_id: source_rc
        kind: reverse_complement
        left_segment_id: source
        right_segment_id: derived
qa:
  require_no_unknown_bases: false
  allow_degenerate_bases: true
output:
  artifact_bundle: artifacts/iupac_reverse_complement
""",
        encoding="utf-8",
    )

    result = run_linear_ssdna_composition(config_path)

    bundle = result.artifact_bundle
    assembled_payload = json.loads((bundle / "assembled_sequence.json").read_text(encoding="utf-8"))
    assert assembled_payload["sequence"]["sequence"] == assembled
    assert assembled_payload["assertions"] == [
        {
            "assertion_id": "source_rc",
            "kind": "reverse_complement",
            "severity": "error",
            "status": "pass",
        }
    ]

    reverse_fasta_lines = (bundle / "sequence.reverse_complement.fa").read_text(encoding="utf-8").splitlines()
    assert reverse_fasta_lines[1] == assembled_reverse_complement
    reverse_genbank = SeqIO.read(bundle / "sequence.reverse_complement.gb", "genbank")
    assert str(reverse_genbank.seq).upper() == assembled_reverse_complement.upper()

    visual = json.loads((bundle / "visual" / "sequence_evidence_map_v1.json").read_text(encoding="utf-8"))
    assert visual["primary_sequence"] == assembled
    assert visual["complement_sequence"] == assembled_complement
