"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/tests/test_scan_codon.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dnadesign.permuter.src.protocols.dms.scan_codon import ScanCodon


CODON_CSV = """codon,amino_acid,fraction,frequency
AAA,K,0.73,33.2
AAG,K,0.27,12.1
AAC,N,0.53,24.4
AAT,N,0.47,21.9
CAA,Q,0.30,12.1
CAG,Q,0.70,27.7
TAA,*,0.99,42.0
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


def _write_table(tmp_path: Path, text: str = CODON_CSV) -> Path:
    p = tmp_path / "codons.csv"
    p.write_text(text, encoding="utf-8")
    return p


def test_scan_codon_region_bounds_and_relative_positions(tmp_path: Path):
    """
    region_codons is 0-based and end-exclusive in CODON units.
    Check absolute vs. relative AA positions and that unrelated codons remain untouched.
    """
    table = _write_table(tmp_path)
    proto = ScanCodon()
    params = {"codon_table": str(table), "region_codons": [1, 3]}  # scan codons 1..2
    # WT: AAA AAA AAA (K K K) → 3 codons
    ref = "A" * 9
    out = list(
        proto.generate(
            ref_entry={"ref_name": "ref", "sequence": ref},
            params=params,
            rng=np.random.default_rng(0),
        )
    )
    assert out  # there should be many AA alts for two codon positions
    # absolute AA pos is 1-based: only positions 2 and 3 are mutated
    aa_pos = {r["aa_pos"] for r in out}
    assert aa_pos == {2, 3}
    # relative AA pos counts from region start → 1 and 2
    aa_pos_rel = {r["aa_pos_rel"] for r in out}
    assert aa_pos_rel == {1, 2}
    # Codon 0 (first) must remain unchanged in all variants
    for r in out:
        s = r["sequence"]
        assert s[:3] == ref[:3]


def test_scan_codon_single_nt_vs_multi_nt_tokening(tmp_path: Path):
    """
    AAA (K) → AAC (N) changes exactly one base (pos3).
    AAA (K) → CAG (Q) changes two bases → nt_pos fields must be absent.
    """
    table = _write_table(tmp_path)
    proto = ScanCodon()
    params = {"codon_table": str(table)}
    ref = "AAA"  # one codon

    outs = list(
        proto.generate(
            ref_entry={"ref_name": "x", "sequence": ref},
            params=params,
            rng=np.random.default_rng(0),
        )
    )
    # Filter to N and Q alts
    n_alts = [r for r in outs if r["aa_alt"] == "N"]
    q_alts = [r for r in outs if r["aa_alt"] == "Q"]
    assert n_alts and q_alts

    # Top N codon per table is AAC → one nt diff at pos3 (1-based)
    n0 = n_alts[0]
    assert n0["codon_wt"] == "AAA" and n0["codon_new"] == "AAC"
    assert n0["nt_pos"] == 3 and n0["nt_wt"] == "A" and n0["nt_alt"] == "C"

    # Top Q codon per table is CAG → multi-nt change → nt_* keys must be absent
    q0 = q_alts[0]
    assert q0["codon_wt"] == "AAA" and q0["codon_new"] == "CAG"
    assert "nt_pos" not in q0 and "nt_wt" not in q0 and "nt_alt" not in q0


def test_scan_codon_skips_wt_codons_missing_from_table(tmp_path: Path):
    """
    If WT codon is a STOP (filtered out) or otherwise absent in the table,
    that codon index should produce no variants.
    """
    table = _write_table(tmp_path)
    proto = ScanCodon()
    params = {"codon_table": str(table)}
    # WT codons: AAA (K), TAA (*) ← stop → loader drops '*'
    ref = "AAATAA"  # two codons, second is stop
    out = list(
        proto.generate(
            ref_entry={"ref_name": "x", "sequence": ref},
            params=params,
            rng=np.random.default_rng(0),
        )
    )
    # All variants should be for the first codon only (aa_pos==1)
    assert out and {r["aa_pos"] for r in out} == {1}


def test_scan_codon_uses_top_ranked_alt_codon(tmp_path: Path):
    """
    For each ALT AA, the new codon should be the top‑weight entry (index 0) in the table.
    Here we check at least one AA (N), whose top is AAC (not AAT).
    """
    table = _write_table(tmp_path)
    proto = ScanCodon()
    params = {"codon_table": str(table)}
    out = list(
        proto.generate(
            ref_entry={"ref_name": "x", "sequence": "AAA"},  # K at pos1
            params=params,
            rng=np.random.default_rng(0),
        )
    )
    n = [r for r in out if r["aa_alt"] == "N"]
    assert n and all(r["codon_new"] == "AAC" for r in n)


def test_scan_codon_validate_cfg_and_sequence_errors(tmp_path: Path):
    # Missing table
    with pytest.raises(ValueError):
        ScanCodon().validate_cfg(params={})
    # Nonexistent path
    with pytest.raises(ValueError):
        ScanCodon().validate_cfg(params={"codon_table": str(tmp_path / "nope.csv")})
    # Bad region_codons shape
    t = _write_table(tmp_path)
    with pytest.raises(ValueError):
        ScanCodon().validate_cfg(params={"codon_table": str(t), "region_codons": [0]})

    # Non-triplet sequence → ValueError in generate()
    proto = ScanCodon()
    proto.validate_cfg(params={"codon_table": str(t)})
    with pytest.raises(ValueError):
        list(
            proto.generate(
                ref_entry={"ref_name": "x", "sequence": "AAAAA"},  # len % 3 != 0
                params={"codon_table": str(t)},
                rng=np.random.default_rng(0),
            )
        )


def test_scan_codon_rejects_malformed_tables(tmp_path: Path):
    # Missing required columns
    bad = tmp_path / "bad.csv"
    bad.write_text("codon,fraction\nAAA,1.0\n", encoding="utf-8")
    proto = ScanCodon()
    with pytest.raises(ValueError):
        proto.validate_cfg(params={"codon_table": str(bad)})
    # Loader path: table present but missing amino_acid → load will blow up during generate
    bad2 = tmp_path / "bad2.csv"
    bad2.write_text("codon,amino_acid\nAAA,K\n", encoding="utf-8")
    proto.validate_cfg(params={"codon_table": str(bad2)})
    # No weights → _load_codon_table raises
    with pytest.raises(ValueError):
        list(
            proto.generate(
                ref_entry={"ref_name": "x", "sequence": "AAA"},
                params={"codon_table": str(bad2)},
                rng=np.random.default_rng(0),
            )
        )

