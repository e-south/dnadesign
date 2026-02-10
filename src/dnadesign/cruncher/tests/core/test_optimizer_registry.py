"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/core/test_optimizer_registry.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import pytest

from dnadesign.cruncher.core.optimizers.registry import get_optimizer, list_optimizer_specs, list_optimizers


def test_optimizer_registry_has_builtins() -> None:
    opts = list_optimizers()
    assert "gibbs_anneal" in opts


def test_optimizer_registry_unknown() -> None:
    with pytest.raises(ValueError):
        get_optimizer("unknown")


def test_optimizer_specs_have_descriptions() -> None:
    specs = {spec.name: spec.description for spec in list_optimizer_specs()}
    assert specs["gibbs_anneal"]


def test_gibbs_optimizer_is_declared_in_gibbs_module() -> None:
    optimizer_factory = get_optimizer("gibbs_anneal")
    assert optimizer_factory.__module__ == "dnadesign.cruncher.core.optimizers.gibbs_anneal"
