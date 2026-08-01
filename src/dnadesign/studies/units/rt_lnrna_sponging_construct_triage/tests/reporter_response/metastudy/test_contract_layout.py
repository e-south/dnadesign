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

from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy import (
    contracts,
    evaluation,
    operator,
)

_METASTUDY_ROOT = Path(__file__).parents[3] / "reporter_response" / "metastudy"
_TEST_ROOT = Path(__file__).parent
_CONTRACTS_ROOT = _METASTUDY_ROOT / "contracts"
_EVALUATION_ROOT = _METASTUDY_ROOT / "evaluation"
_OPERATOR_ROOT = _METASTUDY_ROOT / "operator"
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
_EXPECTED_EVALUATION_MODULES = {"__init__.py", "evidence.py", "readiness.py", "selection.py"}
_EVALUATION_LINE_BUDGETS = {
    "__init__.py": 35,
    "evidence.py": 240,
    "readiness.py": 250,
    "selection.py": 480,
}
_EXPECTED_OPERATOR_MODULES = {
    "__init__.py",
    "__main__.py",
    "cli.py",
    "persistence.py",
    "regeneration.py",
    "state.py",
}
_OPERATOR_LINE_BUDGETS = {
    "__init__.py": 60,
    "__main__.py": 30,
    "cli.py": 150,
    "persistence.py": 160,
    "regeneration.py": 320,
    "state.py": 330,
}
_MATERIALIZE_TEST_ROOT = _TEST_ROOT / "materialize"
_EXPECTED_MATERIALIZE_TEST_MODULES = {
    "__init__.py",
    "_support.py",
    "test_identity.py",
    "test_profiles.py",
    "test_service.py",
    "test_temporal.py",
}
_MATERIALIZE_TEST_LINE_BUDGETS = {
    "__init__.py": 20,
    "_support.py": 340,
    "test_identity.py": 380,
    "test_profiles.py": 370,
    "test_service.py": 150,
    "test_temporal.py": 150,
}
_OPERATOR_TEST_ROOT = _TEST_ROOT / "operator"
_EXPECTED_OPERATOR_TEST_MODULES = {
    "__init__.py",
    "_support.py",
    "test_cli.py",
    "test_persistence.py",
    "test_regeneration.py",
    "test_state.py",
}
_OPERATOR_TEST_LINE_BUDGETS = {
    "__init__.py": 20,
    "_support.py": 200,
    "test_cli.py": 100,
    "test_persistence.py": 120,
    "test_regeneration.py": 380,
    "test_state.py": 200,
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


def test_evaluation_package_has_one_semantic_module_per_owner() -> None:
    assert not (_METASTUDY_ROOT / "evaluation.py").exists()
    assert {path.name for path in _EVALUATION_ROOT.glob("*.py")} == _EXPECTED_EVALUATION_MODULES
    for filename, line_budget in _EVALUATION_LINE_BUDGETS.items():
        line_count = len((_EVALUATION_ROOT / filename).read_text(encoding="utf-8").splitlines())
        assert line_count <= line_budget, f"{filename} exceeds its {line_budget}-line architecture budget"


def test_evaluation_facade_exposes_only_supported_operations() -> None:
    assert set(evaluation.__all__) == {
        "decision_evidence_payload",
        "decision_from_readiness",
        "evaluate_metastudy",
        "reevaluate_evidence_projection",
        "readiness_from_live_bridge",
        "readiness_from_receipt",
    }


def test_operator_package_has_one_semantic_module_per_owner() -> None:
    assert not (_METASTUDY_ROOT / "operator.py").exists()
    assert {path.name for path in _OPERATOR_ROOT.glob("*.py")} == _EXPECTED_OPERATOR_MODULES
    for filename, line_budget in _OPERATOR_LINE_BUDGETS.items():
        line_count = len((_OPERATOR_ROOT / filename).read_text(encoding="utf-8").splitlines())
        assert line_count <= line_budget, f"{filename} exceeds its {line_budget}-line architecture budget"


def test_operator_facade_exposes_only_supported_operator_names() -> None:
    assert set(operator.__all__) == {
        "LiveStateValidation",
        "RegenerationResult",
        "build_parser",
        "main",
        "regenerate_metastudy",
        "validate_live_source_controlled_state",
        "validate_source_controlled_state",
        "write_source_controlled_state",
    }


def test_materialization_tests_are_split_by_behavior_owner() -> None:
    assert not (_TEST_ROOT / "test_materialize.py").exists()
    assert {path.name for path in _MATERIALIZE_TEST_ROOT.glob("*.py")} == _EXPECTED_MATERIALIZE_TEST_MODULES
    for filename, line_budget in _MATERIALIZE_TEST_LINE_BUDGETS.items():
        line_count = len((_MATERIALIZE_TEST_ROOT / filename).read_text(encoding="utf-8").splitlines())
        assert line_count <= line_budget, f"{filename} exceeds its {line_budget}-line test-owner budget"


def test_operator_tests_are_split_by_behavior_owner() -> None:
    assert not (_TEST_ROOT / "test_operator.py").exists()
    assert {path.name for path in _OPERATOR_TEST_ROOT.glob("*.py")} == _EXPECTED_OPERATOR_TEST_MODULES
    for filename, line_budget in _OPERATOR_TEST_LINE_BUDGETS.items():
        line_count = len((_OPERATOR_TEST_ROOT / filename).read_text(encoding="utf-8").splitlines())
        assert line_count <= line_budget, f"{filename} exceeds its {line_budget}-line test-owner budget"


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
