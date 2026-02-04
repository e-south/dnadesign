"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/tests/test_case_preserve_dna.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import numpy as np

from dnadesign.permuter.src.protocols.dms.scan_dna import ScanDNA


def test_scan_dna_preserves_case():
    proto = ScanDNA()
    proto.validate_cfg(params={"regions": []})
    orig = "aCgT"  # mixed case
    out = list(
        proto.generate(
            ref_entry={"ref_name": "x", "sequence": orig},
            params={"regions": []},
            rng=np.random.default_rng(0),
        )
    )
    assert out
    for rec in out:
        seq = rec["sequence"]
        pos = rec["nt_pos"] - 1
        # Unchanged flanks
        assert seq[:pos] == orig[:pos]
        assert seq[pos + 1 :] == orig[pos + 1 :]
        # Mutated char keeps original casing at that position
        assert seq[pos].isupper() == orig[pos].isupper()
        # And base actually changed (ignoring case)
        assert seq[pos].upper() != orig[pos].upper()
