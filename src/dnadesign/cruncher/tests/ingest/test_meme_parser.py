"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/ingest/test_meme_parser.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dnadesign.cruncher.io.parsers.meme import parse_meme_text, select_meme_motif


def _meme_text() -> str:
    return (
        "MEME version 5.5.3\n"
        "ALPHABET= ACGT\n"
        "Background letter frequencies (from dataset):\n"
        "A 0.25 C 0.25 G 0.25 T 0.25\n"
        "\n"
        "MOTIF MEME-1 cusR\n"
        "letter-probability matrix: alength= 4 w= 3 nsites= 4 E= 1e-4\n"
        "0.2 0.3 0.4 0.1\n"
        "0.1 0.1 0.7 0.1\n"
        "0.25 0.25 0.25 0.25\n"
        "log-odds matrix: alength= 4 w= 3 nsites= 4 E= 1e-4\n"
        "1 0 -1 -2\n"
        "0 2 -1 -1\n"
        "-1 -1 2 0\n"
        "Motif 1 sites sorted by position p-value\n"
        "----------------------------------------\n"
        "sequence_name start p-value site\n"
        "seq1 10 1e-4 ACG\n"
        "seq2 20 0.01 ATG\n"
        "Motif 1 sites in BLOCKS format\n"
        "seq1 (10) ACG\n"
        "seq2 (20) ATG\n"
        "\n"
        "MOTIF MEME-2 lexA\n"
        "letter-probability matrix: alength= 4 w= 2 nsites= 3 E= 1e-5\n"
        "0.25 0.25 0.25 0.25\n"
        "0.2 0.3 0.1 0.4\n"
        "Motif 2 sites in BLOCKS format\n"
        "seq3 (5) AC\n"
        "seq4 (7) AT\n"
    )


def test_parse_meme_text_multi_motif_blocks() -> None:
    result = parse_meme_text(_meme_text(), Path("demo.txt"))
    assert result.meta.background_freqs == (0.25, 0.25, 0.25, 0.25)
    assert len(result.motifs) == 2
    first, second = result.motifs
    assert first.width == 3
    assert second.width == 2
    assert len(first.prob_matrix) == 3
    assert len(second.prob_matrix) == 2
    assert first.log_odds_matrix is not None
    assert len(first.block_sites) == 2
    assert first.block_sites[0].pvalue == pytest.approx(1e-4)
    assert first.block_sites[1].pvalue == pytest.approx(0.01)
    assert second.block_sites[0].pvalue is None


def test_meme_blocks_header_is_ignored() -> None:
    text = _meme_text().replace(
        "Motif 1 sites in BLOCKS format\n",
        "Motif 1 sites in BLOCKS format\nBL   MOTIF cusR width=3 seqs=4\n",
    )
    result = parse_meme_text(text, Path("demo.txt"))
    assert len(result.motifs) == 2
    assert len(result.motifs[0].block_sites) == 2


def test_meme_blocks_terminator_is_ignored() -> None:
    text = _meme_text() + "\n//\n"
    result = parse_meme_text(text, Path("demo.txt"))
    assert len(result.motifs) == 2


def test_select_meme_motif_by_index() -> None:
    result = parse_meme_text(_meme_text(), Path("demo.txt"))
    motif = select_meme_motif(result, file_stem="demo", selector=2, path=Path("demo.txt"))
    assert motif.motif_name == "lexA"


def test_select_meme_motif_ambiguous_name_match() -> None:
    text = _meme_text().replace("MEME-2 lexA", "MEME-2 cusR")
    result = parse_meme_text(text, Path("cusR.txt"))
    with pytest.raises(ValueError):
        select_meme_motif(result, file_stem="cusR", selector=None, path=Path("cusR.txt"))


def test_meme_invalid_alphabet_raises() -> None:
    text = _meme_text().replace("ALPHABET= ACGT", "ALPHABET= ACGU")
    with pytest.raises(ValueError):
        parse_meme_text(text, Path("bad.txt"))


def test_meme_invalid_blocks_sequence_raises() -> None:
    text = _meme_text().replace("seq1 (10) ACG", "seq1 (10) ACN")
    with pytest.raises(ValueError):
        parse_meme_text(text, Path("bad_blocks.txt"))
