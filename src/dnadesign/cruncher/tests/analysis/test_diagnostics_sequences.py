"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/analysis/test_diagnostics_sequences.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pandas as pd
import pytest

from dnadesign.cruncher.analysis.diagnostics import summarize_sampling_diagnostics


def test_unique_fraction_prefers_canonical_sequence() -> None:
    df = pd.DataFrame(
        {
            "phase": ["draw", "draw"],
            "sequence": ["ATGC", "GCAT"],
            "canonical_sequence": ["ATGC", "ATGC"],
        }
    )
    diagnostics = summarize_sampling_diagnostics(
        trace_idata=None,
        sequences_df=df,
        elites_df=pd.DataFrame(),
        elites_hits_df=None,
        tf_names=["tf1"],
        optimizer={"kind": "gibbs_anneal"},
        optimizer_stats={},
        sample_meta={"dsdna_canonicalize": True},
        trace_required=False,
    )
    seq_metrics = diagnostics["metrics"]["sequences"]
    assert seq_metrics["unique_sequences"] == 1
    assert abs(seq_metrics["unique_fraction"] - 0.5) < 1e-6


def test_diagnostics_warn_when_unique_successes_below_min() -> None:
    df = pd.DataFrame(
        {
            "phase": ["draw"],
            "sequence": ["ATGC"],
            "score_tf1": [0.2],
        }
    )
    diagnostics = summarize_sampling_diagnostics(
        trace_idata=None,
        sequences_df=df,
        elites_df=pd.DataFrame(),
        elites_hits_df=None,
        tf_names=["tf1"],
        optimizer={"kind": "gibbs_anneal"},
        optimizer_stats={"unique_successes": 1},
        sample_meta={"dsdna_canonicalize": False, "early_stop": {"require_min_unique": True, "min_unique": 2}},
        trace_required=False,
    )
    warnings = diagnostics.get("warnings") or []
    assert any("unique successes" in warning for warning in warnings)


def test_diagnostics_includes_pvalue_cache_stats() -> None:
    diagnostics = summarize_sampling_diagnostics(
        trace_idata=None,
        sequences_df=None,
        elites_df=pd.DataFrame(),
        elites_hits_df=None,
        tf_names=["tf1"],
        optimizer={"kind": "gibbs_anneal"},
        optimizer_stats={},
        sample_meta={"pvalue_cache": {"hits": 2, "misses": 1, "maxsize": 256, "currsize": 3}},
        trace_required=False,
    )
    metrics = diagnostics["metrics"]
    assert metrics["pvalue_cache"] == {"hits": 2, "misses": 1, "maxsize": 256, "currsize": 3}


def test_gibbs_diagnostics_do_not_emit_swap_warning() -> None:
    diagnostics = summarize_sampling_diagnostics(
        trace_idata=None,
        sequences_df=pd.DataFrame({"sequence": []}),
        elites_df=pd.DataFrame(),
        elites_hits_df=None,
        tf_names=["tf1"],
        optimizer={"kind": "gibbs_anneal"},
        optimizer_stats={"swap_attempts": 0, "swap_acceptance_rate": 0.0},
        optimizer_kind="gibbs_anneal",
        trace_required=False,
    )
    warnings = diagnostics.get("warnings") or []
    assert not any("Swap acceptance" in warning for warning in warnings)
    optimizer_metrics = diagnostics["metrics"]["optimizer"]
    assert "swap_acceptance_rate" not in optimizer_metrics


def test_unique_fraction_uses_canonical_column_without_identity_remap(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame(
        {
            "phase": ["draw", "draw"],
            "sequence": ["ATGC", "GCAT"],
            "canonical_sequence": ["ATGC", "ATGC"],
        }
    )

    def _identity_key_should_not_run(seq: str, *, bidirectional: bool) -> str:
        raise AssertionError(f"identity_key should not be called, saw {seq}, {bidirectional}")

    monkeypatch.setattr("dnadesign.cruncher.analysis.diagnostics.identity_key", _identity_key_should_not_run)

    diagnostics = summarize_sampling_diagnostics(
        trace_idata=None,
        sequences_df=df,
        elites_df=pd.DataFrame(),
        elites_hits_df=None,
        tf_names=["tf1"],
        optimizer={"kind": "gibbs_anneal"},
        optimizer_stats={},
        sample_meta={"dsdna_canonicalize": True},
        trace_required=False,
    )

    seq_metrics = diagnostics["metrics"]["sequences"]
    assert seq_metrics["unique_sequences_canonical"] == 1
    assert seq_metrics["unique_sequences_raw"] == 2
