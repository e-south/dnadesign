"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/tests/test_scan_codon.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from pathlib import Path

import numpy as np

from dnadesign.permuter.src.protocols.scan_codon import ScanCodon

CODON_CSV = """codon,amino_acid,fraction,frequency
AAA,K,0.73,33.2
AAG,K,0.27,12.1
AAC,N,0.53,24.4
AAT,N,0.47,21.9
CAA,Q,0.3,12.1
CAG,Q,0.7,27.7
"""


def test_scan_codon_generates_by_usage_ranking(tmp_path: Path):
    table = tmp_path / "codon.csv"
    table.write_text(CODON_CSV, encoding="utf-8")
    proto = ScanCodon()
    params = {"codon_table": str(table)}
    proto.validate_cfg(params=params)
    # WT: AAA AAG  (K K), length 6
    out = list(
        proto.generate(
            ref_entry={"ref_name": "x", "sequence": "AAAAAG"},
            params=params,
            rng=np.random.default_rng(0),
        )
    )
    assert out, "expected some codon-substitution variants"
    for rec in out:
        assert len(rec["sequence"]) == 6
        assert "aa_pos" in rec and rec["aa_pos"] >= 1
        # nt_pos field present only when a single-nucleotide change
        if "nt_pos" in rec:
            assert rec["nt_wt"] in "ACGT" and rec["nt_alt"] in "ACGT"
