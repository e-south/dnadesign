"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/tests/test_utils_mutsets.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import pytest

from dnadesign.permuter.src.protocols.multisite_select.utils import (
    pairwise_hamming_from_mutmaps,
    parse_aa_combo_to_map,
)


def test_parse_aa_combo_to_map_basic():
    m = parse_aa_combo_to_map("G16F|L17I|N21H")
    assert m == {16: "F", 17: "I", 21: "H"}


@pytest.mark.parametrize("bad", ["", " ", "16F", "G16", "G16FI", "X-1Z"])
def test_parse_raises_on_malformed(bad):
    if bad.strip() == "":
        assert parse_aa_combo_to_map(bad) == {}
    else:
        with pytest.raises(ValueError):
            parse_aa_combo_to_map(bad)


def test_pairwise_hamming_from_mutmaps():
    a = parse_aa_combo_to_map("A10V|B20C")
    b = parse_aa_combo_to_map("A10V|B20C")  # identical -> 0
    c = parse_aa_combo_to_map("A10V|B20Y")  # one pos differs -> 1
    d = parse_aa_combo_to_map(
        "A10V|C30K"
    )  # one shared, one disjoint -> 1 (C30K vs none)
    e = parse_aa_combo_to_map(
        "D40E"
    )  # all disjoint from a -> 3 mutated positions union: {10,20,40} -> 2 diffs
    dists = pairwise_hamming_from_mutmaps([a, b, c, d, e])
    # sanity checks on a vs others
    # order is (0,1),(0,2),(0,3),(0,4),...
    assert dists[0] == 0  # a vs b
    assert dists[1] == 1  # a vs c (B20C vs B20Y)
    assert dists[2] == 1  # a vs d (C30K vs none)
    assert dists[3] == 2  # a vs e (positions 10,20 vs 40 => two diffs)
