"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/test_optimizer_registry.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import pytest

from dnadesign.cruncher.core.optimizers.registry import get_optimizer, list_optimizer_specs, list_optimizers


def test_optimizer_registry_has_builtins() -> None:
    opts = list_optimizers()
    assert "gibbs" in opts
    assert "pt" in opts


def test_optimizer_registry_unknown() -> None:
    with pytest.raises(ValueError):
        get_optimizer("unknown")


def test_optimizer_specs_have_descriptions() -> None:
    specs = {spec.name: spec.description for spec in list_optimizer_specs()}
    assert specs["gibbs"]
    assert specs["pt"]
