"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/twist_handoff/_assertions.py

Assertions for human-reviewable Eco1 RT GenBank handoffs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from Bio import SeqIO

_PROTECTED_FEATURE_LABELS = {
    "Protected motif context: NAxxH",
    "Protected motif context: YADD",
    "Protected motif context: VTG",
    "Protected: direct DNA/RNA contacts (<=5 A)",
    "Protected: Wang thumb-contact track",
    "Protected: primer-recognition context (255-311)",
    "Protected: conserved/core positions",
}


def assert_reviewable_genbank_records(
    *,
    genbank_paths: tuple[Path, ...],
    manifest_rows: list[dict[str, Any]],
) -> None:
    """Assert that exported records expose mutations and protected context plainly."""

    assert len(genbank_paths) == 8
    for path in genbank_paths:
        record = SeqIO.read(path, "genbank")
        assert len(record.seq) == 963
        assert record.description.startswith("Eco1 reverse transcriptase variant with ")
        assert record.annotations["source"] == "synthetic DNA construct"
        assert len([feature for feature in record.features if feature.type == "CDS"]) == 1
        mutation_features = [feature for feature in record.features if feature.type == "variation"]
        manifest_row = next(row for row in manifest_rows if row["sequence_id"] == record.id)
        mutation_labels = [str(feature.qualifiers["label"][0]) for feature in mutation_features]

        assert len(mutation_features) == len(manifest_row["mutation_tokens"])
        assert mutation_labels == manifest_row["mutation_tokens"]
        assert [str(feature.qualifiers["standard_name"][0]) for feature in mutation_features] == mutation_labels
        assert all(len(feature.location) == 3 for feature in mutation_features)
        assert all(
            str(feature.qualifiers["note"][0]).startswith(f"{label}: reference ")
            for feature, label in zip(mutation_features, mutation_labels, strict=True)
        )

        assert [feature.type for feature in record.features[:3]] == ["gene", "CDS", "misc_feature"]
        summary = record.features[2]
        assert summary.qualifiers["label"] == [f"{len(mutation_labels)} amino-acid substitutions"]
        assert summary.qualifiers["note"][1] == f"Changes: {', '.join(mutation_labels)}"
        assert record.annotations["comment"].startswith(
            f"AMINO-ACID SUBSTITUTIONS ({len(mutation_labels)}): {', '.join(mutation_labels)}."
        )
        assert record.features[3 : 3 + len(mutation_labels)] == mutation_features

        feature_labels = {
            str(feature.qualifiers.get("label", [""])[0])
            for feature in record.features
            if feature.type == "misc_feature"
        }
        assert "Review: Wang alpha-1 interface (4-16)" in feature_labels
        assert "Review: Wang R13 interface residue" in feature_labels
        assert _PROTECTED_FEATURE_LABELS.issubset(feature_labels)
        assert not any(label.startswith("protected_context:") for label in feature_labels)
        assert not any("5-10" in label for label in feature_labels)
        assert len(record.features) == len(mutation_labels) + 12
        feature_notes = [str(note) for feature in record.features for note in feature.qualifiers.get("note", [])]
        assert any("RT-msDNA assembly state: not_established" in note for note in feature_notes)


__all__ = ["assert_reviewable_genbank_records"]
