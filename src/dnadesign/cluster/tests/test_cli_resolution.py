"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cluster/tests/test_cli_resolution.py

CLI resolution contracts for cluster preset and method-param merging.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest
import typer

from dnadesign.cluster.src.cli.resolution import resolve_fit_method_params


def test_resolve_fit_method_params_allows_cli_overrides_for_preset_keys() -> None:
    params = resolve_fit_method_params(
        {},
        ["resolution=0.9"],
        preset_name="method.leiden.fine",
    )

    assert params["resolution"] == "0.9"


def test_resolve_fit_method_params_rejects_workspace_overlap_with_preset_keys() -> None:
    with pytest.raises(typer.BadParameter, match="overlap with the selected preset"):
        resolve_fit_method_params(
            {"method_params": {"resolution": 0.5}},
            [],
            preset_name="method.leiden.fine",
        )
