"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/reporter_response/metastudy/operator/test_checkout.py

Meta-study checkout ownership contract tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy import (
    MetastudyContractError,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy.operator.checkout import (
    active_dnadesign_checkout,
    require_active_dnadesign_checkout,
)


def test_active_checkout_is_the_checkout_containing_the_imported_package() -> None:
    checkout = active_dnadesign_checkout()

    assert checkout == next(
        parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file()
    )
    assert (checkout / "src/dnadesign/__init__.py").resolve() == Path(__import__("dnadesign").__file__).resolve()


def test_checkout_contract_defaults_to_the_active_checkout() -> None:
    assert require_active_dnadesign_checkout(None) == active_dnadesign_checkout()


def test_checkout_contract_rejects_a_missing_checkout(tmp_path: Path) -> None:
    with pytest.raises(MetastudyContractError, match="does not exist"):
        require_active_dnadesign_checkout(tmp_path / "missing")


def test_checkout_contract_rejects_a_foreign_checkout(tmp_path: Path) -> None:
    foreign = tmp_path / "dnadesign"
    (foreign / "src/dnadesign").mkdir(parents=True)
    (foreign / "pyproject.toml").write_text("[project]\nname = 'dnadesign'\n", encoding="utf-8")
    (foreign / "src/dnadesign/__init__.py").write_text("", encoding="utf-8")

    with pytest.raises(MetastudyContractError, match="must be the active source checkout"):
        require_active_dnadesign_checkout(foreign)
