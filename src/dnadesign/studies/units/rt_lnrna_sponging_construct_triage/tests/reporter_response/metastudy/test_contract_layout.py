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
    evidence_projection,
    operator,
)

_METASTUDY_ROOT = Path(__file__).parents[3] / "reporter_response" / "metastudy"
_TEST_ROOT = Path(__file__).parent
_CONTRACTS_ROOT = _METASTUDY_ROOT / "contracts"
_EVALUATION_ROOT = _METASTUDY_ROOT / "evaluation"
_OPERATOR_ROOT = _METASTUDY_ROOT / "operator"
_ACQUISITION_PROJECTION_ROOT = _METASTUDY_ROOT / "acquisition_projection"
_EVIDENCE_PROJECTION_ROOT = _METASTUDY_ROOT / "evidence_projection"
_SENSITIVITY_COVERAGE_ROOT = _METASTUDY_ROOT / "sensitivity_coverage"
_EXPECTED_CONTRACT_MODULES = {
    "__init__.py",
    "_values.py",
    "candidate.py",
    "decision.py",
    "decision_codec.py",
    "materialization.py",
    "objective.py",
    "profile.py",
    "profile_identity.py",
    "protocol.py",
    "sensitivity.py",
}
_LINE_BUDGETS = {
    "__init__.py": 90,
    "_values.py": 90,
    "candidate.py": 150,
    "decision.py": 180,
    "decision_codec.py": 240,
    "materialization.py": 400,
    "objective.py": 90,
    "profile.py": 170,
    "profile_identity.py": 70,
    "protocol.py": 270,
    "sensitivity.py": 50,
}
_EXPECTED_EVALUATION_MODULES = {
    "__init__.py",
    "candidate.py",
    "comparability.py",
    "evidence.py",
    "readiness.py",
    "selection.py",
}
_EVALUATION_LINE_BUDGETS = {
    "__init__.py": 35,
    "candidate.py": 190,
    "comparability.py": 40,
    "evidence.py": 240,
    "readiness.py": 250,
    "selection.py": 320,
}
_EXPECTED_OPERATOR_MODULES = {
    "__init__.py",
    "__main__.py",
    "checkout.py",
    "cli.py",
    "persistence.py",
    "regeneration.py",
    "state.py",
}
_OPERATOR_LINE_BUDGETS = {
    "__init__.py": 60,
    "__main__.py": 30,
    "checkout.py": 80,
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
    "test_reference_profiles.py",
    "test_service.py",
    "test_temporal.py",
}
_MATERIALIZE_TEST_LINE_BUDGETS = {
    "__init__.py": 20,
    "_support.py": 340,
    "test_identity.py": 380,
    "test_profiles.py": 370,
    "test_reference_profiles.py": 210,
    "test_service.py": 150,
    "test_temporal.py": 150,
}
_OPERATOR_TEST_ROOT = _TEST_ROOT / "operator"
_EXPECTED_OPERATOR_TEST_MODULES = {
    "__init__.py",
    "_support.py",
    "test_checkout.py",
    "test_cli.py",
    "test_persistence.py",
    "test_regeneration.py",
    "test_state.py",
}
_OPERATOR_TEST_LINE_BUDGETS = {
    "__init__.py": 20,
    "_support.py": 200,
    "test_checkout.py": 100,
    "test_cli.py": 150,
    "test_persistence.py": 120,
    "test_regeneration.py": 380,
    "test_state.py": 200,
}
_ACQUISITION_PROJECTION_LINE_BUDGETS = {
    "__init__.py": 30,
    "_values.py": 60,
    "building.py": 220,
    "contracts.py": 230,
    "serialization.py": 130,
}
_EVIDENCE_PROJECTION_LINE_BUDGETS = {
    "__init__.py": 25,
    "_values.py": 60,
    "audit_parsing.py": 60,
    "contracts.py": 130,
    "parsing.py": 330,
}
_EVIDENCE_PROJECTION_ALLOWED_SIBLING_IMPORTS = {
    "_values.py": set(),
    "audit_parsing.py": {"_values", "contracts"},
    "contracts.py": {"_values"},
    "parsing.py": {"_values", "audit_parsing", "contracts"},
}
_SENSITIVITY_COVERAGE_LINE_BUDGETS = {
    "__init__.py": 60,
    "_values.py": 40,
    "building.py": 120,
    "contracts.py": 180,
    "serialization.py": 230,
    "validation.py": 180,
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


def test_contract_modules_do_not_depend_on_evaluation_implementations() -> None:
    for path in _CONTRACTS_ROOT.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.level >= 2
            and node.module is not None
            and node.module.startswith("evaluation")
        ]
        assert not imports, f"{path.name} must not depend on evaluation implementations"


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


def test_acquisition_projection_has_one_semantic_module_per_owner() -> None:
    assert not (_METASTUDY_ROOT / "acquisition_projection.py").exists()
    assert {path.name for path in _ACQUISITION_PROJECTION_ROOT.glob("*.py")} == set(
        _ACQUISITION_PROJECTION_LINE_BUDGETS
    )
    for filename, line_budget in _ACQUISITION_PROJECTION_LINE_BUDGETS.items():
        line_count = len((_ACQUISITION_PROJECTION_ROOT / filename).read_text(encoding="utf-8").splitlines())
        assert line_count <= line_budget, f"{filename} exceeds its {line_budget}-line architecture budget"


def test_evidence_projection_has_one_semantic_module_per_owner() -> None:
    assert not (_METASTUDY_ROOT / "evidence_projection.py").exists()
    assert {path.name for path in _EVIDENCE_PROJECTION_ROOT.glob("*.py")} == set(_EVIDENCE_PROJECTION_LINE_BUDGETS)
    for filename, line_budget in _EVIDENCE_PROJECTION_LINE_BUDGETS.items():
        line_count = len((_EVIDENCE_PROJECTION_ROOT / filename).read_text(encoding="utf-8").splitlines())
        assert line_count <= line_budget, f"{filename} exceeds its {line_budget}-line architecture budget"


def test_evidence_projection_facade_exposes_only_supported_names() -> None:
    assert set(evidence_projection.__all__) == {
        "ProfileContentProjection",
        "ProfileEvidenceProjection",
        "ProfileProvenanceProjection",
        "parse_profile_evidence_projection",
        "profile_source_identity_projection",
    }


def test_evidence_projection_dependencies_follow_the_semantic_owner_graph() -> None:
    for filename, allowed in _EVIDENCE_PROJECTION_ALLOWED_SIBLING_IMPORTS.items():
        path = _EVIDENCE_PROJECTION_ROOT / filename
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        sibling_imports = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module is not None
        }
        assert sibling_imports <= allowed, (
            f"{filename} has reverse or undeclared dependencies: {sibling_imports - allowed}"
        )


def test_metastudy_production_modules_import_evidence_projection_owners() -> None:
    facade = _EVIDENCE_PROJECTION_ROOT / "__init__.py"
    for path in _METASTUDY_ROOT.rglob("*.py"):
        if path == facade:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        facade_imports = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module is not None
            and node.module.endswith("evidence_projection")
        ]
        assert not facade_imports, f"{path.relative_to(_METASTUDY_ROOT)} must import the evidence owner leaf"


def test_sensitivity_coverage_has_one_semantic_module_per_owner() -> None:
    assert not (_METASTUDY_ROOT / "sensitivity_coverage.py").exists()
    assert {path.name for path in _SENSITIVITY_COVERAGE_ROOT.glob("*.py")} == set(_SENSITIVITY_COVERAGE_LINE_BUDGETS)
    for filename, line_budget in _SENSITIVITY_COVERAGE_LINE_BUDGETS.items():
        line_count = len((_SENSITIVITY_COVERAGE_ROOT / filename).read_text(encoding="utf-8").splitlines())
        assert line_count <= line_budget, f"{filename} exceeds its {line_budget}-line architecture budget"


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
