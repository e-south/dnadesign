"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/tests/review_outputs/test_pwm_retention.py

Tests for bidirectional tetR PWM trim-retention selection.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.studies.units.retron_hairpin_design.review_outputs.pwm_retention import (
    PwmMotifOccurrence,
    load_meme_information_bits,
    select_best_retained_span,
)

from ..support.paths import repo_root_from


def test_dual_site_retention_selector_finds_mild_and_stronger_trim_windows() -> None:
    repo_root = repo_root_from(__file__)
    meme_path = (
        repo_root
        / "src"
        / "dnadesign"
        / "cruncher"
        / "workspaces"
        / "demo_monotypic_tetr"
        / "outputs"
        / "artifacts"
        / "meme"
        / "tetR__westmann_tetr_mitomi__tetR.meme"
    )
    motif_bits = load_meme_information_bits(Path(meme_path))
    occurrences = (
        PwmMotifOccurrence(motif_instance_id="tetR:0:17:+:1", start_0=0, end_0=17, strand="+", occurrence_rank=1),
        PwmMotifOccurrence(motif_instance_id="tetR:2:19:-:2", start_0=2, end_0=19, strand="-", occurrence_rank=2),
    )

    mild = select_best_retained_span(
        parent_length=19, retained_length=15, motif_bits=motif_bits, occurrences=occurrences
    )
    stronger = select_best_retained_span(
        parent_length=19,
        retained_length=12,
        motif_bits=motif_bits,
        occurrences=occurrences,
    )

    assert (mild.start_0, mild.end_0) == (2, 17)
    assert mild.sequence_from("CTCTATATCTGATATAGAG") == "CTATATCTGATATAG"
    assert round(mild.retained_information_fraction, 6) == 0.964248
    assert [round(value, 6) for value in mild.retained_bits_by_occurrence] == [6.785073, 6.785073]

    assert (stronger.start_0, stronger.end_0) == (3, 15)
    assert stronger.sequence_from("CTCTATATCTGATATAGAG") == "TATATCTGATAT"
    assert round(stronger.retained_information_fraction, 6) == 0.867985
    assert [round(value, 6) for value in stronger.retained_bits_by_occurrence] == [6.306051, 5.909354]
