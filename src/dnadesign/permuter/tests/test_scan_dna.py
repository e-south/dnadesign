"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/tests/test_scan_dna.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pytest

from dnadesign.permuter.src.protocols.dms.scan_dna import ScanDNA


def test_scan_dna_counts_and_tokens():
    proto = ScanDNA()
    proto.validate_cfg(params={"regions": []})
    seq = "ACGT"  # length 4
    out = list(
        proto.generate(
            ref_entry={"ref_name": "x", "sequence": seq},
            params={"regions": []},
            rng=np.random.default_rng(0),
        )
    )
    # 4 positions * 3 alts each = 12
    assert len(out) == 12
    # check token and fields consistency
    for rec in out:
        mods = rec["modifications"]
        assert isinstance(mods, list) and mods and "nt pos=" in mods[0]
        assert rec["nt_pos"] >= 1 and rec["nt_pos"] <= len(seq)
        assert rec["nt_wt"] in "ACGT" and rec["nt_alt"] in "ACGT"
        assert rec["nt_wt"] != rec["nt_alt"]


def test_scan_dna_regions_subset_and_counts():
    """
    Two non-overlapping regions: only those positions should be mutated.
    Regions are 0-based [start, end) in nt units.
    """
    proto = ScanDNA()
    seq = "ACGTAC"  # len = 6; positions (1-based): 1..6
    params = {"regions": [[1, 3], [4, 5]]}  # mutate nt indices {1,2,4}
    out = list(
        proto.generate(
            ref_entry={"ref_name": "x", "sequence": seq},
            params=params,
            rng=np.random.default_rng(0),
        )
    )
    # each position gets 3 alts
    assert len(out) == (2 + 1) * 3 == 9
    allowed_pos_1b = {2, 3, 5}
    assert {r["nt_pos"] for r in out}.issubset(allowed_pos_1b)
    # verify no change outside regions
    for r in out:
        s = r["sequence"]
        i = r["nt_pos"] - 1
        assert s[:i] == seq[:i] and s[i + 1 :] == seq[i + 1 :]
        assert s[i].upper() == r["nt_alt"] and s[i].upper() != r["nt_wt"]


@pytest.mark.parametrize(
    "regions",
    [
        "notalist",  # wrong type
        [[1, 1]],  # empty interval (end must be > start)
        [[-1, 2]],  # negative start
        [[0, 999]],  # end beyond length
        [[0]],  # malformed pair
        123,  # wrong type
    ],
)
def test_scan_dna_rejects_bad_regions(regions):
    proto = ScanDNA()
    with pytest.raises(ValueError):
        list(
            proto.generate(
                ref_entry={"ref_name": "x", "sequence": "ACGT"},
                params={"regions": regions},
                rng=np.random.default_rng(0),
            )
        )


def test_scan_dna_raises_on_non_dna_symbols():
    proto = ScanDNA()
    with pytest.raises(ValueError):
        list(
            proto.generate(
                ref_entry={"ref_name": "x", "sequence": "ACGX"},
                params={"regions": []},
                rng=np.random.default_rng(0),
            )
        )


def test_scan_dna_multiple_regions_accumulate_without_overlap():
    """
    Multiple regions add up; no extra edits outside them.
    """
    proto = ScanDNA()
    seq = "AAAAAA"
    params = {"regions": [[0, 2], [3, 6]]}  # mutate pos {1,2,4,5,6} (1-based)
    out = list(
        proto.generate(
            ref_entry={"ref_name": "x", "sequence": seq},
            params=params,
            rng=np.random.default_rng(0),
        )
    )
    assert len(out) == (2 + 3) * 3 == 15
    assert {r["nt_pos"] for r in out} == {1, 2, 4, 5, 6}
