from __future__ import annotations

from dnadesign.densegen.src.core.artifacts.ids import hash_pwm_motif, hash_tfbs_id


def test_hash_tfbs_id_is_deterministic() -> None:
    a = hash_tfbs_id(motif_id="M1", sequence="ACGT", scoring_backend="fimo", matched_start=1, matched_stop=4)
    b = hash_tfbs_id(motif_id="M1", sequence="ACGT", scoring_backend="fimo", matched_start=1, matched_stop=4)
    assert a == b


def test_hash_tfbs_id_changes_with_inputs() -> None:
    base = hash_tfbs_id(motif_id="M1", sequence="ACGT", scoring_backend="fimo", matched_start=1, matched_stop=4)
    diff_seq = hash_tfbs_id(motif_id="M1", sequence="TGCA", scoring_backend="fimo", matched_start=1, matched_stop=4)
    diff_match = hash_tfbs_id(motif_id="M1", sequence="ACGT", scoring_backend="fimo", matched_start=2, matched_stop=5)
    assert base != diff_seq
    assert base != diff_match


def test_hash_pwm_motif_changes_with_matrix() -> None:
    m1 = hash_pwm_motif(
        motif_label="lexA",
        matrix=[{"A": 0.7, "C": 0.1, "G": 0.1, "T": 0.1}],
        background={"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
        source_kind="pwm_meme",
    )
    m2 = hash_pwm_motif(
        motif_label="lexA",
        matrix=[{"A": 0.6, "C": 0.2, "G": 0.1, "T": 0.1}],
        background={"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
        source_kind="pwm_meme",
    )
    assert m1 != m2
