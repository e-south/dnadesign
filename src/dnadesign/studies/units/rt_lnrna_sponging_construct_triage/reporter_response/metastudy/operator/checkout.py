"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reporter_response/metastudy/operator/checkout.py

Active-checkout ownership for Dnadesign meta-study inputs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import dnadesign

from ..contracts._values import MetastudyContractError


def active_dnadesign_checkout() -> Path:
    """Return the checkout that contains the imported ``dnadesign`` package."""

    package_file = getattr(dnadesign, "__file__", None)
    if package_file is None:
        raise MetastudyContractError("cannot resolve the active Dnadesign source checkout")
    package_root = Path(package_file).resolve().parent
    checkout = package_root.parent.parent
    if not (checkout / "pyproject.toml").is_file() or package_root != checkout / "src/dnadesign":
        raise MetastudyContractError("active Dnadesign package is not loaded from a source checkout")
    return checkout


def require_active_dnadesign_checkout(value: Path | None) -> Path:
    """Resolve an optional checkout argument and reject cross-checkout reads."""

    active = active_dnadesign_checkout()
    if value is None:
        return active
    supplied = Path(value).expanduser()
    if not supplied.exists():
        raise MetastudyContractError(f"Dnadesign checkout does not exist: {supplied}")
    if not supplied.is_dir():
        raise MetastudyContractError(f"Dnadesign checkout is not a directory: {supplied}")
    resolved = supplied.resolve()
    if resolved != active:
        raise MetastudyContractError(
            f"Dnadesign checkout must be the active source checkout: expected={active} observed={resolved}"
        )
    return active


__all__ = ["active_dnadesign_checkout", "require_active_dnadesign_checkout"]
