"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_exit_codes_lock.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from dnadesign.opal.src.utils import ExitCodes


def test_exit_codes_has_contract_violation():
    assert hasattr(ExitCodes, "CONTRACT_VIOLATION")
