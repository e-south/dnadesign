"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/sae_window_summary/__init__.py

Eco1 SAE window-summary materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .models import MaterializedSaeWindowSummary, WindowSpec
from .pipeline import materialize_sae_window_summary

__all__ = ["MaterializedSaeWindowSummary", "WindowSpec", "materialize_sae_window_summary"]
