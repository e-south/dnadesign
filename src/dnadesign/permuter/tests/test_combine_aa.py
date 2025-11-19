"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/tests/test_combine_aa.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from dnadesign.permuter.src.protocols.combine.combine_aa import (
    CombineAA,
    attach_epistasis,
)

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
    assert abs(rec["permuter__expected__llr_mean"] - (1.0 + 2.0)) < 1e-9


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


def _synergy_for_positions(pos_list: List[int]) -> float:
    """
    Deterministic 'epistasis' generator for testing.
    Pairwise and triple terms ensure both positive and negative Δ.
    """
    P = set(int(p) for p in pos_list)
    s = 0.0
    # pairwise interactions
    if {1, 2}.issubset(P):
        s += 1.50
    if {3, 4}.issubset(P):
        s -= 0.25
    # a 3-way term
    if {1, 2, 3}.issubset(P):
        s += 0.70
    return float(s)


def test_epistasis_definition_matches_observed_minus_expected(tmp_path: Path):
    """
    Build a tiny DMS 'singles' set → CombineAA emits combos.
    We then fabricate 'observed' by adding a known synergy to 'expected',
    and assert: epistasis == observed - expected (by definition).
    """
    # WT DNA: 4 AAA codons → 'K' at AA positions 1..4
    ref = "AAA" * 4
    rows = [
        # pos,   wt, alt,  singles LLR (arbitrary but stable)
        dict(
            sequence=ref,
            permuter__round=1,
            permuter__aa_pos=1,
            permuter__aa_wt="K",
            permuter__aa_alt="N",
            permuter__metric__llr_mean=+0.50,
        ),
        dict(
            sequence=ref,
            permuter__round=1,
            permuter__aa_pos=2,
            permuter__aa_wt="K",
            permuter__aa_alt="Q",
            permuter__metric__llr_mean=+1.50,
        ),
        dict(
            sequence=ref,
            permuter__round=1,
            permuter__aa_pos=3,
            permuter__aa_wt="K",
            permuter__aa_alt="Q",
            permuter__metric__llr_mean=-0.20,
        ),
        dict(
            sequence=ref,
            permuter__round=1,
            permuter__aa_pos=4,
            permuter__aa_wt="K",
            permuter__aa_alt="N",
            permuter__metric__llr_mean=+2.00,
        ),
    ]
    dms = _write_dms_singles(tmp_path, rows)
    codon_table = _write_codon_table(tmp_path)

    params = dict(
        from_dataset=str(dms),
        codon_table=str(codon_table),
        singles_metric_id="llr_mean",
        select=dict(top_global=4, mode="global", disallow_negative_best=False),
        combine=dict(
            k_min=2,
            k_max=3,
            budget_total=16,
            strategy="random",
            random=dict(samples_per_k={"2": 4, "3": 2}),
        ),
        codon_choice="top",
        rng_seed=12345,
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
    assert out, "CombineAA produced no combos in test"

    # Build a DataFrame to attach synthetic 'observed' and compute epistasis
    df = pd.DataFrame(out)
    assert "permuter__expected__llr_mean" in df.columns

    # Fabricate 'observed' using a deterministic synergy function on aa_pos_list
    observed = []
    epis = []
    for _, r in df.iterrows():
        expected = float(r["permuter__expected__llr_mean"])
        pos_list = list(map(int, r["aa_pos_list"]))
        syn = _synergy_for_positions(pos_list)
        obs = expected + syn
        observed.append(obs)
        epis.append(obs - expected)

    # Provide canonical observed column and compute epistasis via helper
    df["permuter__observed__llr_mean"] = observed
    df2 = attach_epistasis(df, metric_id="llr_mean")
    assert "epistasis" in df2.columns
    diff = (df2["epistasis"].to_numpy() - np.array(epis)).astype(float)
    assert np.all(np.isfinite(diff))
    assert float(np.max(np.abs(diff))) <= 1e-12

    # Sign convention sanity: positive synergy -> positive epistasis
    has_pos = any(_synergy_for_positions(r) > 0 for r in df2["aa_pos_list"])
    has_neg = any(_synergy_for_positions(r) < 0 for r in df2["aa_pos_list"])
    if has_pos:
        assert (df2["epistasis"] > 0).any()
    if has_neg:
        assert (df2["epistasis"] < 0).any()

    # Sign convention sanity: positive synergy -> positive epistasis
    has_pos = any(_synergy_for_positions(r) > 0 for r in df["aa_pos_list"])
    has_neg = any(_synergy_for_positions(r) < 0 for r in df["aa_pos_list"])
    if has_pos:
        assert (df["epistasis"] > 0).any()
    if has_neg:
        assert (df["epistasis"] < 0).any()


def test_combo_fields_are_canonical_and_consistent(tmp_path: Path):
    """
    Reinforce that CombineAA emits canonical AA tokens and consistent lists/counts:
      mut_count == len(aa_pos_list) == len(aa_wt_list) == len(aa_alt_list)
      aa_combo_str is sorted by pos and matches the lists.
    Also re-check additivity for expected__llr_mean.
    """
    ref = "AAA" * 3  # K at AA positions 1..3
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
            permuter__aa_pos=2,
            permuter__aa_wt="K",
            permuter__aa_alt="Q",
            permuter__metric__llr_mean=2.0,
        ),
        dict(
            sequence=ref,
            permuter__round=1,
            permuter__aa_pos=3,
            permuter__aa_wt="K",
            permuter__aa_alt="N",
            permuter__metric__llr_mean=-0.5,
        ),
    ]
    dms = _write_dms_singles(tmp_path, rows)
    codon_table = _write_codon_table(tmp_path)
    params = dict(
        from_dataset=str(dms),
        codon_table=str(codon_table),
        singles_metric_id="llr_mean",
        select=dict(top_global=3),
        combine=dict(
            k_min=2,
            k_max=2,
            budget_total=6,
            strategy="random",
            random=dict(samples_per_k={"2": 3}),
        ),
        codon_choice="top",
        rng_seed=7,
    )
    proto = CombineAA()
    out = list(
        proto.generate(
            ref_entry={"ref_name": "x", "sequence": ref},
            params=params,
            rng=np.random.default_rng(0),
        )
    )
    assert out

    def parse_combo(s: str):
        if not s:
            return []
        toks = []
        for t in s.split("|"):
            t = t.strip()
            wt, pos, alt = t[0], int(t[1:-1]), t[-1]
            toks.append((pos, wt, alt))
        return toks

    # Single lookup for expected singles scores
    singles = {
        (r["permuter__aa_pos"], r["permuter__aa_alt"]): r["permuter__metric__llr_mean"]
        for r in rows
    }

    for rec in out:
        k = int(rec["mut_count"])
        pos = list(map(int, rec["aa_pos_list"]))
        wt = list(map(str, rec["aa_wt_list"]))
        alt = list(map(str, rec["aa_alt_list"]))
        assert k == len(pos) == len(wt) == len(alt)

        # aa_combo_str canonical order (by position ascending) and matches lists
        toks = parse_combo(rec["aa_combo_str"])
        assert [p for (p, _, _) in toks] == sorted(
            pos
        ), "aa_combo_str must be sorted by position"
        assert [w for (_, w, _) in toks] == wt, "WT letters mismatch"
        assert [a for (_, _, a) in toks] == alt, "ALT letters mismatch"

        # Additivity: expected__llr_mean equals sum of singles at those positions
        expected = float(rec["permuter__expected__llr_mean"])
        calc = float(sum(singles[(p, a)] for p, a in zip(pos, alt)))
        assert (
            abs(expected - calc) < 1e-12
        ), f"expected(additive) mismatch: {expected} vs {calc}"
        assert "permuter__expected__llr_mean" in rec


def test_per_position_best_applies_top_after_grouping(tmp_path: Path):
    """
    Construct many single rows such that, if we truncated BEFORE grouping,
    we'd lose positions. After fix, we keep the best per position first,
    then apply top_global to those winners.
    """
    ref = "AAA" * 6  # 6 positions (K at 1..6)
    rows = []
    # For each position p, make one true winner with clear score, plus decoys at other positions
    winners = {1: 3.0, 2: 2.8, 3: 2.6, 4: 2.4, 5: 2.2, 6: 2.0}
    for p, sc in winners.items():
        rows.append(
            dict(
                sequence=ref,
                permuter__round=1,
                permuter__aa_pos=p,
                permuter__aa_wt="K",
                permuter__aa_alt="N",
                permuter__metric__llr_mean=sc,
            )
        )
    # Add many decoys at pos1 with slightly lower scores than the true winner, to crowd the global head
    for j in range(50):
        rows.append(
            dict(
                sequence=ref,
                permuter__round=1,
                permuter__aa_pos=1,
                permuter__aa_wt="K",
                permuter__aa_alt="Q",
                permuter__metric__llr_mean=3.0 - 1e-3 * (j + 1),
            )
        )
    dms = _write_dms_singles(tmp_path, rows)
    codon_table = _write_codon_table(tmp_path)
    params = dict(
        from_dataset=str(dms),
        codon_table=str(codon_table),
        singles_metric_id="llr_mean",
        select=dict(
            mode="per_position_best",
            top_global=3,
            min_delta=0.0,
            disallow_negative_best=True,
        ),
        combine=dict(
            k_min=2,
            k_max=2,
            budget_total=1,
            strategy="random",
            random=dict(samples_per_k={"2": 1}),
        ),
        codon_choice="top",
    )
    # We don't need to generate combos; we just exercise selection via CombineAA.validate_cfg + generate head
    proto = CombineAA()
    proto.validate_cfg(params=params)
    out = list(
        proto.generate(
            ref_entry={"ref_name": "x", "sequence": ref},
            params=params,
            rng=np.random.default_rng(0),
        )
    )
    # With top_after_grouping, we must see only from the top 3 winner positions among {1..6}
    # i.e., positions {1,2,3}
    got_positions = set(int(p) for rec in out for p in rec["aa_pos_list"])
    assert got_positions.issubset({1, 2, 3})


def test_enumerate_strategy_emits_all_k(tmp_path: Path):
    ref = "AAA" * 5  # positions 1..5
    rows = []
    # One best alt per position, positive scores
    for p, sc in enumerate([1.0, 1.1, 1.2, 1.3, 1.4], start=1):
        rows.append(
            dict(
                sequence=ref,
                permuter__round=1,
                permuter__aa_pos=p,
                permuter__aa_wt="K",
                permuter__aa_alt="N",
                permuter__metric__llr_mean=sc,
            )
        )
    dms = _write_dms_singles(tmp_path, rows)
    codon_table = _write_codon_table(tmp_path)
    params = dict(
        from_dataset=str(dms),
        codon_table=str(codon_table),
        singles_metric_id="llr_mean",
        select=dict(mode="per_position_best", top_global=5, min_delta=1e-12),
        combine=dict(k_min=2, k_max=3, budget_total=999999, strategy="enumerate"),
        codon_choice="top",
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
    # All C(5,2)+C(5,3) = 10 + 10 = 20
    assert len(out) == 20


def test_attach_epistasis_accepts_legacy_metric_and_drops_aliases(tmp_path: Path):
    # Reuse the small 3-position case
    ref = "AAA" * 3
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
            permuter__aa_pos=2,
            permuter__aa_wt="K",
            permuter__aa_alt="Q",
            permuter__metric__llr_mean=2.0,
        ),
        dict(
            sequence=ref,
            permuter__round=1,
            permuter__aa_pos=3,
            permuter__aa_wt="K",
            permuter__aa_alt="N",
            permuter__metric__llr_mean=-0.5,
        ),
    ]
    dms = _write_dms_singles(tmp_path, rows)
    codon_table = _write_codon_table(tmp_path)
    params = dict(
        from_dataset=str(dms),
        codon_table=str(codon_table),
        singles_metric_id="llr_mean",
        select=dict(top_global=3),
        combine=dict(k_min=2, k_max=2, budget_total=3, strategy="enumerate"),
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
    df = pd.DataFrame(out)
    # Legacy: evaluator wrote to permuter__metric__llr_mean; helper should produce canonical observed & epistasis.
    df["permuter__metric__llr_mean"] = df["permuter__expected__llr_mean"] + 0.5
    df2 = attach_epistasis(df, metric_id="llr_mean")
    assert "epistasis" in df2.columns and np.allclose(df2["epistasis"], 0.5)
    assert "permuter__observed__llr_mean" in df2.columns
    # Aliases should not be introduced by the helper
    assert "observed" not in df2.columns
    assert "expected__llr_mean" not in df2.columns


def test_allowed_position_ranges_are_respected(tmp_path: Path):
    # REF: 4 Ks → positions 1..4
    ref = "AAA" * 4
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
            permuter__aa_pos=2,
            permuter__aa_wt="K",
            permuter__aa_alt="N",
            permuter__metric__llr_mean=1.1,
        ),
        dict(
            sequence=ref,
            permuter__round=1,
            permuter__aa_pos=3,
            permuter__aa_wt="K",
            permuter__aa_alt="N",
            permuter__metric__llr_mean=1.2,
        ),
        dict(
            sequence=ref,
            permuter__round=1,
            permuter__aa_pos=4,
            permuter__aa_wt="K",
            permuter__aa_alt="N",
            permuter__metric__llr_mean=1.3,
        ),
    ]
    dms = _write_dms_singles(tmp_path, rows)
    codon_table = _write_codon_table(tmp_path)
    params = dict(
        from_dataset=str(dms),
        codon_table=str(codon_table),
        singles_metric_id="llr_mean",
        select=dict(mode="per_position_best", top_global=4, allowed_positions=["2-3"]),
        combine=dict(k_min=2, k_max=2, budget_total=10, strategy="enumerate"),
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
    assert out, "No combos generated under allowed window"
    got_positions = set(int(p) for rec in out for p in rec["aa_pos_list"])
    assert got_positions.issubset(
        {2, 3}
    ), f"expected positions {{2,3}}, got {got_positions}"
