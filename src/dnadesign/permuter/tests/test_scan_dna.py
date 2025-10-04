"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/tests/test_scan_dna.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import numpy as np

from dnadesign.permuter.src.protocols.scan_dna import ScanDNA


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
