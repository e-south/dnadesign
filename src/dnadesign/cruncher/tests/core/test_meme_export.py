"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/tests/core/test_meme_export.py

Tests for minimal MEME export helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dnadesign.cruncher.core.pwm import PWM
from dnadesign.cruncher.io.meme_export import build_minimal_meme_text, sanitize_meme_id, write_minimal_meme_motif


def test_sanitize_meme_id_normalizes_invalid_characters_and_empty_ids() -> None:
    assert sanitize_meme_id(" tetR/1 ") == "tetR_1"
    assert sanitize_meme_id(" / ") == "motif"


def test_build_minimal_meme_text_normalizes_background_and_writes_file(tmp_path: Path) -> None:
    pwm = PWM(
        name="tetR",
        matrix=[
            [0.7, 0.1, 0.1, 0.1],
            [0.25, 0.25, 0.25, 0.25],
        ],
    )

    motif_id, text = build_minimal_meme_text(
        pwm,
        motif_id=" tetR/1 ",
        background=(1.0, 1.0, 2.0, 0.0),
    )

    assert motif_id == "tetR_1"
    assert "Background letter frequencies:" in text
    assert "A 0.25 C 0.25 G 0.5 T 0" in text
    assert "MOTIF tetR_1" in text

    out_path = tmp_path / "tetR.meme"
    assert write_minimal_meme_motif(pwm, out_path, motif_id=" tetR/1 ") == "tetR_1"
    assert out_path.read_text(encoding="utf-8").startswith("MEME version 4")


@pytest.mark.parametrize(
    ("background", "message"),
    [
        ((0.25, 0.25, 0.25), "4 values"),
        ((0.0, 0.0, 0.0, 0.0), "sum to > 0"),
    ],
)
def test_build_minimal_meme_text_rejects_invalid_background(background: tuple[float, ...], message: str) -> None:
    pwm = PWM(name="tetR", matrix=[[0.25, 0.25, 0.25, 0.25]])

    with pytest.raises(ValueError, match=message):
        build_minimal_meme_text(pwm, background=background)


def test_build_minimal_meme_text_rejects_non_acgt_alphabet_order() -> None:
    pwm = PWM(name="tetR", matrix=[[0.25, 0.25, 0.25, 0.25]], alphabet=("A", "T", "C", "G"))

    with pytest.raises(ValueError, match="A,C,G,T alphabet order"):
        build_minimal_meme_text(pwm)
