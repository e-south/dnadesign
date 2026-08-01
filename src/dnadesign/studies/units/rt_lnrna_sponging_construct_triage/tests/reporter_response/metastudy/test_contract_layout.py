"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/reporter_response/metastudy/test_contract_layout.py

Architecture contracts for the reporter-response meta-study contract package.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import ast
from pathlib import Path

from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy import contracts

_METASTUDY_ROOT = Path(__file__).parents[3] / "reporter_response" / "metastudy"
_CONTRACTS_ROOT = _METASTUDY_ROOT / "contracts"
_EXPECTED_CONTRACT_MODULES = {
    "__init__.py",
    "_values.py",
    "decision.py",
    "materialization.py",
    "profile.py",
    "protocol.py",
}
_LINE_BUDGETS = {
    "__init__.py": 90,
    "_values.py": 90,
    "decision.py": 550,
    "materialization.py": 400,
    "profile.py": 170,
    "protocol.py": 270,
}


def test_contract_package_has_one_semantic_module_per_owner() -> None:
    assert not (_METASTUDY_ROOT / "contracts.py").exists()
    assert {path.name for path in _CONTRACTS_ROOT.glob("*.py")} == _EXPECTED_CONTRACT_MODULES
    for filename, line_budget in _LINE_BUDGETS.items():
        line_count = len((_CONTRACTS_ROOT / filename).read_text(encoding="utf-8").splitlines())
        assert line_count <= line_budget, f"{filename} exceeds its {line_budget}-line architecture budget"


def test_contract_facade_exposes_only_supported_contract_names() -> None:
    assert set(contracts.__all__) == {
        "DECISION_CONTRACT_ID",
        "DEFAULT_OBJECTIVE_READINESS",
        "DEFAULT_PROTOCOL",
        "PROTOCOL_ID",
        "CandidateEvaluation",
        "EvidenceReadiness",
        "GrowthPhaseStratum",
        "MaterializationAttemptReceipt",
        "MaterializationBlocker",
        "MaterializationOmission",
        "MetastudyContractError",
        "MetastudyDecision",
        "MetastudyProtocol",
        "ObjectiveReadiness",
        "ProfileAuditArtifact",
        "ProfileEvidence",
        "ReaderRecordIdentity",
        "SensitivityEvaluation",
        "canonical_digest",
        "decision_to_dict",
        "materialization_attempt_payload",
        "objective_readiness_from_payload",
        "protocol_digest",
        "validate_decision_payload",
    }


def test_production_leaves_import_contract_owners_not_facade() -> None:
    exceptions = {_METASTUDY_ROOT / "__init__.py"}
    for path in _METASTUDY_ROOT.glob("*.py"):
        if path in exceptions:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        facade_imports = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module == "contracts"
        ]
        assert not facade_imports, f"{path.name} must import the specific contract owner module"
