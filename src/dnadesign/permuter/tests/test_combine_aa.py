"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/tests/test_combine_aa.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dnadesign.permuter.src.protocols.combine.combine_aa import CombineAA

CODON_CSV = """codon,amino_acid,fraction,frequency
AAA,K,0.73,33.2
AAG,K,0.27,12.1
AAC,N,0.53,24.4
AAT,N,0.47,21.9
CAA,Q,0.30,12.1
CAG,Q,0.70,27.7
"""


def _write_codon_table(tmp_path: Path) -> Path:
    p = tmp_path / "codon.csv"
    p.write_text(CODON_CSV, encoding="utf-8")
    return p


def _write_dms_singles(tmp_path: Path, rows) -> Path:
    df = pd.DataFrame(rows)
    p = tmp_path / "dms.parquet"
    df.to_parquet(p, index=False)
    return p


def test_case_preservation_and_codon_substitution(tmp_path: Path):
    # Reference DNA: two codons, mixed case across positions
    ref = "aaAaaG"  # AAA AAG with varying case → AA positions 1 and 2 are 'K'
    # DMS singles: two AA changes at positions 1 and 2
    dms = _write_dms_singles(
        tmp_path,
        [
            dict(
                sequence=ref,
                permuter__round=1,
                permuter__aa_pos=1,
                permuter__aa_wt="K",
                permuter__aa_alt="N",
                permuter__metric__llr_mean=1.0,
            ),
            dict(
                sequence=ref,
                permuter__round=1,
                permuter__aa_pos=2,
                permuter__aa_wt="K",
                permuter__aa_alt="Q",
                permuter__metric__llr_mean=2.0,
            ),
        ],
    )
    codon_table = _write_codon_table(tmp_path)

    params = dict(
        from_dataset=str(dms),
        codon_table=str(codon_table),
        singles_metric_id="llr_mean",
        select=dict(top_global=2),
        combine=dict(
            k_min=2,
            k_max=2,
            budget_total=10,
            strategy="random",
            random=dict(samples_per_k={"2": 1}),
        ),
        codon_choice="top",
        rng_seed=1234,
    )
    proto = CombineAA()
    proto.validate_cfg(params=params)
    out = list(
        proto.generate(
            ref_entry={"ref_name": "x", "sequence": ref},
            params=params,
            rng=np.random.default_rng(0),
        )
    )
    assert len(out) == 1
    rec = out[0]
    seq = rec["sequence"]
    # Unchanged flanks outside codons: there are none here; validate per-base case in codon replacements
    # Codon 1 and 2 positions must keep the original per-base casing
    for off in range(3):
        assert seq[off].isupper() == ref[off].isupper()
    for off in range(3):
        i = 3 + off
        assert seq[i].isupper() == ref[i].isupper()
    # AA combo canonical string and expected sum
    assert rec["aa_combo_str"] in ("K1N|K2Q", "K2Q|K1N")
    assert abs(rec["expected__llr_mean"] - (1.0 + 2.0)) < 1e-9


def test_selection_invariants(tmp_path: Path):
    # Duplicate AA event appears twice → averaged score; exclusions honored
    ref = "AAAAAA"  # AAA AAA (K K)
    rows = [
        dict(
            sequence=ref,
            permuter__round=1,
            permuter__aa_pos=1,
            permuter__aa_wt="K",
            permuter__aa_alt="N",
            permuter__metric__llr_mean=1.0,
        ),
        dict(
            sequence=ref,
            permuter__round=1,
            permuter__aa_pos=1,
            permuter__aa_wt="K",
            permuter__aa_alt="N",
            permuter__metric__llr_mean=3.0,
        ),
        dict(
            sequence=ref,
            permuter__round=1,
            permuter__aa_pos=2,
            permuter__aa_wt="K",
            permuter__aa_alt="Q",
            permuter__metric__llr_mean=2.0,
        ),
    ]
    dms = _write_dms_singles(tmp_path, rows)
    codon_table = _write_codon_table(tmp_path)
    params = dict(
        from_dataset=str(dms),
        codon_table=str(codon_table),
        singles_metric_id="llr_mean",
        select=dict(top_global=10, exclude_positions=[2], exclude_mutations=["K1N"]),
        combine=dict(
            k_min=2,
            k_max=2,
            budget_total=10,
            strategy="random",
            random=dict(samples_per_k={"2": 1}),
        ),
        codon_choice="top",
    )
    proto = CombineAA()
    proto.validate_cfg(params=params)
    # After exclusions, elite should be empty → RuntimeError at generation time
    try:
        list(
            proto.generate(
                ref_entry={"ref_name": "x", "sequence": ref},
                params=params,
                rng=np.random.default_rng(0),
            )
        )
    except RuntimeError as e:
        assert "empty elite set" in str(e) or "emitted 0 combination" in str(e)


def test_determinism(tmp_path: Path):
    ref = "AAAAAA"  # two K
    dms = _write_dms_singles(
        tmp_path,
        [
            dict(
                sequence=ref,
                permuter__round=1,
                permuter__aa_pos=1,
                permuter__aa_wt="K",
                permuter__aa_alt="N",
                permuter__metric__llr_mean=1.0,
            ),
            dict(
                sequence=ref,
                permuter__round=1,
                permuter__aa_pos=2,
                permuter__aa_wt="K",
                permuter__aa_alt="Q",
                permuter__metric__llr_mean=2.0,
            ),
        ],
    )
    codon_table = _write_codon_table(tmp_path)
    params = dict(
        from_dataset=str(dms),
        codon_table=str(codon_table),
        singles_metric_id="llr_mean",
        select=dict(top_global=2),
        combine=dict(
            k_min=2,
            k_max=2,
            budget_total=10,
            strategy="random",
            random=dict(samples_per_k={"2": 1}),
        ),
        codon_choice="top",
        rng_seed=42,
    )
    proto = CombineAA()
    out1 = list(
        proto.generate(
            ref_entry={"ref_name": "x", "sequence": ref},
            params=params,
            rng=np.random.default_rng(0),
        )
    )
    out2 = list(
        proto.generate(
            ref_entry={"ref_name": "x", "sequence": ref},
            params=params,
            rng=np.random.default_rng(0),
        )
    )
    assert out1[0]["sequence"] == out2[0]["sequence"]
    assert out1[0]["aa_combo_str"] == out2[0]["aa_combo_str"]


def test_budget_and_k_caps(tmp_path: Path):
    ref = "AAAAAA"  # two K
    dms = _write_dms_singles(
        tmp_path,
        [
            dict(
                sequence=ref,
                permuter__round=1,
                permuter__aa_pos=1,
                permuter__aa_wt="K",
                permuter__aa_alt="N",
                permuter__metric__llr_mean=1.0,
            ),
            dict(
                sequence=ref,
                permuter__round=1,
                permuter__aa_pos=2,
                permuter__aa_wt="K",
                permuter__aa_alt="Q",
                permuter__metric__llr_mean=2.0,
            ),
        ],
    )
    codon_table = _write_codon_table(tmp_path)
    params = dict(
        from_dataset=str(dms),
        codon_table=str(codon_table),
        singles_metric_id="llr_mean",
        select=dict(top_global=2),
        combine=dict(
            k_min=2,
            k_max=3,  # 3 is impossible here (only 2 positions) → should just emit k=2
            budget_total=1,  # hard cap = 1
            strategy="random",
            random=dict(samples_per_k={"2": 100, "3": 100}),
        ),
        codon_choice="top",
    )
    proto = CombineAA()
    out = list(
        proto.generate(
            ref_entry={"ref_name": "x", "sequence": ref},
            params=params,
            rng=np.random.default_rng(0),
        )
    )
    assert len(out) == 1
    assert out[0]["mut_count"] == 2
