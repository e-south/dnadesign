"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_selection_contracts.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.opal.src.core.selection_contracts import (
    RESERVED_SELECTION_PARAM_KEYS,
    extract_selection_plugin_params,
    resolve_selection_objective_mode,
    resolve_selection_tie_handling,
)
from dnadesign.opal.src.core.utils import OpalError


def test_resolve_selection_objective_mode_normalizes_case() -> None:
    mode = resolve_selection_objective_mode({"objective_mode": "  MAXIMIZE "})
    assert mode == "maximize"


def test_resolve_selection_objective_mode_requires_field() -> None:
    with pytest.raises(ValueError, match="selection.params.objective_mode is required"):
        resolve_selection_objective_mode({})


def test_resolve_selection_tie_handling_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="selection.params.tie_handling must be one of"):
        resolve_selection_tie_handling({"tie_handling": "random"})


def test_selection_contract_parser_supports_opal_error_type() -> None:
    with pytest.raises(OpalError, match="selection.params.objective_mode must be maximize\\|minimize"):
        resolve_selection_objective_mode({"objective_mode": "sideways"}, error_cls=OpalError)


def test_extract_selection_plugin_params_excludes_reserved_keys() -> None:
    params = {
        "top_k": 10,
        "tie_handling": "competition_rank",
        "objective_mode": "maximize",
        "score_ref": "sfxi_v1/sfxi",
        "uncertainty_ref": "sfxi_v1/sfxi",
        "exclude_already_labeled": True,
        "alpha": 0.5,
        "beta": 1.0,
    }
    plugin_params = extract_selection_plugin_params(params)
    assert plugin_params == {"alpha": 0.5, "beta": 1.0}


def test_reserved_selection_param_keys_are_complete_for_runtime_keys() -> None:
    assert "top_k" in RESERVED_SELECTION_PARAM_KEYS
    assert "tie_handling" in RESERVED_SELECTION_PARAM_KEYS
    assert "objective_mode" in RESERVED_SELECTION_PARAM_KEYS
    assert "score_ref" in RESERVED_SELECTION_PARAM_KEYS
    assert "uncertainty_ref" in RESERVED_SELECTION_PARAM_KEYS
    assert "exclude_already_labeled" in RESERVED_SELECTION_PARAM_KEYS
