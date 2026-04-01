"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/ingest/test_ddg_table_parser.py

Tests for delta-delta-G table parsing and conversion.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dnadesign.cruncher.io.parsers.ddg_table import ddg_to_probability_matrix, parse_ddg_table


def _write_westmann_ddg(path: Path) -> None:
    path.write_text(
        "PO\tA\tT\tC\tG\n"
        "1\t0.164072\t0.519334\t-0.0468541\t0.22966\n"
        "2\t0.642569\t0.164072\t0.188965\t0.943314\n"
        "3\t1.04481\t0.558693\t0.164072\t0.848264\n"
        "4\t0.959646\t0.164072\t2.17536\t0.718959\n"
        "5\t0.164072\t1.19894\t1.79469\t1.56685\n"
        "6\t1.46463\t0.164072\t1.74384\t1.54845\n"
        "7\t0.966397\t1.07557\t0.164072\t1.72883\n"
        "8\t0.164072\t0.504312\t0.755195\t0.148902\n"
        "9\t0.0876645\t0.164072\t0\t0.0156484\n"
        "10\t0.289722\t0.164072\t0.00707563\t0.843474\n"
        "11\t1.77568\t1.3447\t2.34804\t0.164072\n"
        "12\t0.164072\t1.72354\t1.62782\t1.49176\n"
        "13\t0.877518\t0.164072\t1.30879\t1.9052\n"
        "14\t0.164072\t0.544642\t0.3387\t1.7537\n"
        "15\t0.540091\t0.8821\t1.0861\t0.164072\n"
        "16\t-0.224147\t0.280358\t0.442769\t0.164072\n"
        "17\t0.31238\t-0.0651059\t0.164072\t-0.303844\n",
        encoding="utf-8",
    )


def test_ddg_to_probability_matrix_prefers_lower_energy_states() -> None:
    matrix = np.array([[0.0, 1.0, 2.0, 3.0]], dtype=float)
    probs = ddg_to_probability_matrix(matrix, temperature_k=298.15)
    assert probs.shape == (1, 4)
    assert np.isclose(probs.sum(axis=1)[0], 1.0)
    assert probs[0, 0] > probs[0, 1] > probs[0, 2] > probs[0, 3]


def test_parse_ddg_table_converts_westmann_table_to_pwm(tmp_path: Path) -> None:
    path = tmp_path / "tetR.tsv"
    _write_westmann_ddg(path)

    pwm = parse_ddg_table(path)

    assert pwm.length == 17
    assert pwm.alphabet == ("A", "C", "G", "T")
    assert np.allclose(pwm.matrix.sum(axis=1), 1.0)
    # Row 1 has the lowest ddG for C in the raw A/T/C/G order.
    assert int(np.argmax(pwm.matrix[0])) == 1
    # Row 16 has the lowest ddG for A.
    assert int(np.argmax(pwm.matrix[15])) == 0
    # Row 17 has the lowest ddG for G.
    assert int(np.argmax(pwm.matrix[16])) == 2


def test_parse_ddg_table_rejects_nonsequential_positions(tmp_path: Path) -> None:
    path = tmp_path / "bad.tsv"
    path.write_text("PO\tA\tT\tC\tG\n1\t0\t0\t0\t0\n3\t0\t0\t0\t0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="position order"):
        parse_ddg_table(path)
