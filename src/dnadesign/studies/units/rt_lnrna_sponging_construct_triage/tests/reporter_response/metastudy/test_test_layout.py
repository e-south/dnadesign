"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/reporter_response/metastudy/test_test_layout.py

Guards the semantic decomposition of reporter-response metastudy tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

SEMANTIC_TEST_MODULES = (
    "acquisition/test_projection.py",
    "contracts/test_candidate.py",
    "contracts/test_decision.py",
    "contracts/test_protocol.py",
    "evidence/test_audits.py",
    "evidence/test_projection.py",
    "publication/test_atomicity.py",
    "publication/test_integrity.py",
    "publication/test_sensitivity.py",
    "readiness/test_readiness.py",
    "selection/test_selection.py",
)
BOUNDED_SUPPORT_MODULES = ("_builders.py", "evidence/_builders.py", "publication/_builders.py")
MAX_SEMANTIC_MODULE_LINES = 500
MAX_SUPPORT_MODULE_LINES = 400


def test_metastudy_tests_use_bounded_semantic_modules() -> None:
    root = Path(__file__).parent
    discovered_semantic_modules = {
        path.relative_to(root).as_posix()
        for directory in ("acquisition", "contracts", "evidence", "publication", "readiness", "selection")
        for path in (root / directory).glob("test_*.py")
    }

    assert not (root / "test_metastudy.py").exists()
    assert discovered_semantic_modules == set(SEMANTIC_TEST_MODULES)
    assert all((root / relative).is_file() for relative in BOUNDED_SUPPORT_MODULES)
    assert all(
        len((root / relative).read_text(encoding="utf-8").splitlines()) <= MAX_SEMANTIC_MODULE_LINES
        for relative in SEMANTIC_TEST_MODULES
    )
    assert all(
        len((root / relative).read_text(encoding="utf-8").splitlines()) <= MAX_SUPPORT_MODULE_LINES
        for relative in BOUNDED_SUPPORT_MODULES
    )
