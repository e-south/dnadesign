"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/tests/test_artifacts_contract.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import os
from pathlib import Path

import pandas as pd
import pytest

ART_DIR = Path(os.environ.get("PERMUTER_TEST_ART_DIR", ""))


COMBINE_DIR = os.environ.get("PERMUTER_TEST_COMBINE_DIR", "")


@pytest.mark.skipif(
    not ART_DIR.exists(), reason="Set PERMUTER_TEST_ART_DIR to a finished run dir"
)
def test_multisite_variants_has_source_id_and_unique_mutsets():
    picks = pd.read_parquet(ART_DIR / "MULTISITE_VARIANTS.parquet")
    # contract: id propagated
    assert "source_id" in picks.columns

    # mutation-set uniqueness: no two picks have identical (pos->alt) sets
    def key(s: str) -> frozenset:
        if not s:
            return frozenset()
        items = []
        for tok in s.split("|"):
            tok = tok.strip()
            alt = tok[-1]
            pos = int(tok[1:-1])
            items.append((pos, alt))
        return frozenset(sorted(items))

    keys = picks["aa_combo_str"].map(key)
    assert keys.is_unique, "duplicate mutation sets found in selected variants"


@pytest.mark.skipif(
    not COMBINE_DIR or not Path(COMBINE_DIR).exists(),
    reason="Set PERMUTER_TEST_COMBINE_DIR to a finished rt_combine run dir",
)
def test_rt_combine_epistasis_equals_observed_minus_expected():
    p = Path(COMBINE_DIR) / "records.parquet"
    assert p.exists(), f"records.parquet not found at {p}"
    df = pd.read_parquet(p)

    # Must have observed and expected
    for col in ("permuter__metric__llr_mean", "permuter__expected__llr_mean"):
        assert col in df.columns, f"missing column: {col}"

    # If epistasis is persisted, it must equal observed-expected within tight tolerance
    if "epistasis" in df.columns:
        obs = df["permuter__metric__llr_mean"].astype(float)
        exp = df["permuter__expected__llr_mean"].astype(float)
        epi = df["epistasis"].astype(float)
        mask = obs.notna() & exp.notna() & epi.notna()
        if mask.any():
            diff = (epi[mask] - (obs[mask] - exp[mask])).abs().max()
            assert (
                float(diff) <= 5e-6
            ), f"epistasis deviates from observed-expected by {float(diff):.3g}"
    # expected_kind, when present, must be 'additive'
    ek = (
        "permuter__expected_kind"
        if "permuter__expected_kind" in df.columns
        else ("expected_kind" if "expected_kind" in df.columns else None)
    )
    if ek is not None:
        assert (df[ek].dropna().astype(str) == "additive").all()
