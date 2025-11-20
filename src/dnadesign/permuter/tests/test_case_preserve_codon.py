"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/tests/test_case_preserve_codon.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from pathlib import Path

import numpy as np

from dnadesign.permuter.src.protocols.dms.scan_codon import ScanCodon

CODON_CSV = """codon,amino_acid,fraction,frequency
AAA,K,0.73,33.2
AAG,K,0.27,12.1
AAC,N,0.53,24.4
AAT,N,0.47,21.9
CAA,Q,0.30,12.1
CAG,Q,0.70,27.7
"""


def test_scan_codon_preserves_case(tmp_path: Path):
    table = tmp_path / "codon.csv"
    table.write_text(CODON_CSV, encoding="utf-8")
    proto = ScanCodon()
    params = {"codon_table": str(table)}
    proto.validate_cfg(params=params)

    # Two codons, mixed case across positions
    orig = "aaAaaG"  # AAA AAG with varying case
    out = list(
        proto.generate(
            ref_entry={"ref_name": "x", "sequence": orig},
            params=params,
            rng=np.random.default_rng(0),
        )
    )
    assert out
    for rec in out:
        seq = rec["sequence"]
        ci = int(rec["codon_index"])
        start = ci * 3
        # Unchanged positions outside the mutated codon
        for k in range(len(orig)):
            if not (start <= k < start + 3):
                assert seq[k] == orig[k]
        # Mutated codon positions keep original per-base casing
        for off in range(3):
            assert seq[start + off].isupper() == orig[start + off].isupper()
