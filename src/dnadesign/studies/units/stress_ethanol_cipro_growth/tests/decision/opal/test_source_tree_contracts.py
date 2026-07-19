"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/test_source_tree_contracts.py

Source-tree boundaries for the stress-study OPAL integration.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.source_evidence import (
    RMF_ROUND0_SOURCE_EVIDENCE_ROOT,
    SFXI_ROUND0_SOURCE_EVIDENCE_ROOT,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.promoter_candidate_bindings import (
    SCHEMA_ID,
    STUDY_ID,
)

STUDY_ROOT = Path("src/dnadesign/studies/units/stress_ethanol_cipro_growth")
OPAL_CAMPAIGNS_ROOT = Path("src/dnadesign/opal/campaigns")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_sha256(value: object) -> str:
    assert isinstance(value, str)
    assert len(value) == 64
    assert all(character in "0123456789abcdef" for character in value)
    return value


def test_candidate_identity_contract_is_owned_at_study_scope() -> None:
    assert SCHEMA_ID == "dnadesign.study.promoter_candidate_bindings.v1"
    assert STUDY_ID == "stress_ethanol_cipro_growth"
    assert (STUDY_ROOT / "promoter_candidate_bindings" / "README.md").is_file()


def test_opal_campaign_root_contains_only_executable_campaigns() -> None:
    assert {path.name for path in OPAL_CAMPAIGNS_ROOT.iterdir() if path.is_dir()} == {
        "demo_gp_ei",
        "demo_gp_topn",
        "demo_rf_sfxi_topn",
        "secg_msrb_greedy",
    }


def test_msrb_has_one_study_protocol_and_one_matching_executable_campaign() -> None:
    protocol_path = STUDY_ROOT / "decision" / "opal" / "multistate_response_behavior" / "protocol.yaml"
    campaign_path = OPAL_CAMPAIGNS_ROOT / "secg_msrb_greedy" / "configs" / "campaign.yaml"
    study_doc = Path("docs/studies/stress_ethanol_cipro_growth/contexts/opal/multistate-response-behavior.md")

    assert protocol_path.is_file()
    assert campaign_path.is_file()
    assert study_doc.is_file()

    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    campaign = yaml.safe_load(campaign_path.read_text(encoding="utf-8"))
    assert protocol["metric"] == {
        "id": "multistate_response_behavior_v1",
        "acronym": "MSRB",
        "score_channel": "behavior_score",
    }
    assert protocol["status"] == "active_learning_probe"
    assert protocol["normalization"]["derivation"] == {
        "response_scale_basis": "reader_joint_bootstrap_sd_of_declared_on_off_response_pairs",
        "signal_scale_basis": "reader_joint_bootstrap_sd_of_each_reference_relative_state",
        "pair_deduplication": "unique_unordered_state_pair_union",
        "reader_joint_bootstrap_draws": 500,
        "scale_quantile": 0.90,
        "quantile_method": "linear",
    }
    assert campaign["campaign"]["slug"] == "secg_msrb_greedy"
    assert campaign["data"]["y_column_name"] == "opal__reader_response_window_vector_v1__y"

    protocol_views = {view["id"]: view["target_mask"] for view in protocol["target_views"]}
    campaign_views = {view["id"]: view for view in campaign["selection_views"]}
    assert set(campaign_views) == set(protocol_views)
    for view_id, target_mask in protocol_views.items():
        objective = campaign_views[view_id]["objective"]
        assert objective["name"] == protocol["metric"]["id"]
        assert objective["params"]["state_ids"] == protocol["assay"]["state_ids"]
        assert objective["params"]["target_mask"] == target_mask
        assert objective["params"]["normalization"] == protocol["normalization"]["values"]
        assert campaign_views[view_id]["selection"]["params"]["score_ref"] == "behavior_score"


def test_msrb_activation_receipt_is_one_way_digest_bound_and_claim_scoped() -> None:
    decision_root = STUDY_ROOT / "decision" / "opal" / "multistate_response_behavior"
    protocol_path = decision_root / "protocol.yaml"
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    pointer = protocol["activation_receipt"]
    assert pointer == {
        "schema_id": "stress_ethanol_cipro_growth.multistate_response_behavior_activation_audit.v1",
        "path": "activation_audit.json",
        "digest_ownership": "receipt_binds_protocol",
    }

    receipt_path = decision_root / pointer["path"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["schema_id"] == pointer["schema_id"]
    assert receipt["schema_version"] == "1"
    assert receipt["study_id"] == "stress_ethanol_cipro_growth"
    assert receipt["protocol_id"] == protocol["protocol_id"]
    assert receipt["objective_id"] == protocol["metric"]["id"]

    decision = receipt["decision"]
    assert decision["semantic_and_mathematical_alignment"]["verdict"] == "go"
    assert decision["active_learning_probe"]["verdict"] == "go"
    assert decision["active_learning_probe"]["authorization_basis"] == "explicit_study_signoff"
    assert decision["prospective_hill_climb_efficacy"]["verdict"] == "unknown"
    assert decision["synthesis_authorization"]["verdict"] == "no_go"

    shadow = receipt["prior_shadow_decision"]
    assert shadow["status"] == "shadow_only"
    assert shadow["campaign_disposition"] == "no_go"
    assert shadow["superseded_claims"] == []
    assert shadow["reconciliation"] == (
        "The shadow no-go remains authoritative for claims of superior hill-climbing and synthesis. "
        "The separately approved active protocol authorizes only a prospectively frozen greedy learning probe."
    )
    shadow_bundle_root = STUDY_ROOT / "workbench" / "outputs" / "multistate_response_behavior_shadow" / "latest"
    packaged_audit_path = (
        STUDY_ROOT
        / "decision"
        / "opal"
        / "response_metastudy"
        / "config"
        / "multistate_response_behavior_adversarial_audit_v1.json"
    )
    expected_evidence_paths = {
        "manifest": shadow_bundle_root / "manifest.json",
        "decision": shadow_bundle_root / "decision.json",
        "normalization": shadow_bundle_root / "normalization.json",
        "independent_adversarial_audit": packaged_audit_path,
    }
    for evidence_id, protocol_key in (
        ("manifest", "shadow_manifest_sha256"),
        ("decision", "shadow_decision_sha256"),
        ("normalization", "shadow_normalization_sha256"),
        ("independent_adversarial_audit", "shadow_adversarial_audit_sha256"),
    ):
        item = shadow["evidence"][evidence_id]
        digest = _require_sha256(item["sha256"])
        path = Path(item["path"])
        assert digest == protocol["evidence"][protocol_key]
        assert path == expected_evidence_paths[evidence_id]
        if evidence_id in {"manifest", "decision", "normalization"}:
            assert item["storage_class"] == "generated_workbench_bundle"
            assert item["verification_owner"] == "multistate_behavior_shadow_bundle_verifier"
        else:
            assert item["storage_class"] == "packaged_source_evidence"
            assert item["verification_owner"] == "activation_receipt_source_tree_contract"
            assert _sha256(path) == digest

    review = receipt["independent_adversarial_review"]
    assert review["reviewer_type"] == "automated_codex_subagent"
    assert review["external_human_peer_review"] is False
    assert review["cryptographic_signature"] is False
    assert review["oracle_reimplementation"]["random_case_count"] == 60_000
    assert review["oracle_reimplementation"]["discrepancy_count"] == 0
    assert review["property_suite"]["collected_test_count"] == 78
    assert review["cardinality_pressure"]["state_counts"] == [2, 4, 8, 16]

    bindings = receipt["source_bindings"]
    assert bindings["algorithm"] == "sha256"
    assert bindings["digest_scope"] == "exact_file_bytes"
    assert bindings["receipt_self_digest"] == "excluded_to_avoid_self_reference"
    entries = bindings["entries"]
    expected_sources = {
        "objective_math": Path("src/dnadesign/opal/src/objectives/multistate_response_behavior_math.py"),
        "objective_plugin": Path("src/dnadesign/opal/src/objectives/multistate_response_behavior_v1.py"),
        "objective_property_tests": Path(
            "src/dnadesign/opal/tests/objectives/test_objective_multistate_response_behavior_v1.py"
        ),
        "generic_objective_definition": Path(
            "src/dnadesign/opal/docs/plugins/objectives/multistate-response-behavior.md"
        ),
        "study_binding": Path("docs/studies/stress_ethanol_cipro_growth/contexts/opal/multistate-response-behavior.md"),
        "active_study_protocol": protocol_path,
        "active_study_protocol_readme": decision_root / "README.md",
        "active_campaign_readme": OPAL_CAMPAIGNS_ROOT / "secg_msrb_greedy" / "README.md",
        "active_campaign_config": OPAL_CAMPAIGNS_ROOT / "secg_msrb_greedy" / "configs" / "campaign.yaml",
        "active_campaign_plot_config": OPAL_CAMPAIGNS_ROOT / "secg_msrb_greedy" / "configs" / "plots.yaml",
        "activation_receipt_verifier": STUDY_ROOT / "tests" / "decision" / "opal" / "test_source_tree_contracts.py",
        "shadow_bundle_verifier": STUDY_ROOT
        / "decision"
        / "opal"
        / "response_metastudy"
        / "runtime"
        / "multistate_behavior_bundle_verification.py",
        "shadow_protocol": STUDY_ROOT
        / "decision"
        / "opal"
        / "response_metastudy"
        / "config"
        / "multistate_response_behavior_shadow_v1.yaml",
    }
    assert {item["role"]: Path(item["path"]) for item in entries} == expected_sources

    paths = [Path(item["path"]) for item in entries]
    assert receipt_path not in paths
    assert len(paths) == len(set(paths))
    assert all(not path.is_absolute() and ".." not in path.parts for path in paths)
    assert protocol_path in paths
    assert all(path.is_file() for path in paths)
    assert all(_require_sha256(item["sha256"]) for item in entries)
    mismatches = {
        path.as_posix(): {"expected": item["sha256"], "actual": _sha256(path)}
        for path, item in zip(paths, entries, strict=True)
        if _sha256(path) != item["sha256"]
    }
    assert mismatches == {}


def test_msrb_study_doc_covers_the_complete_evidence_path() -> None:
    path = Path("docs/studies/stress_ethanol_cipro_growth/contexts/opal/multistate-response-behavior.md")
    text = path.read_text(encoding="utf-8")

    for required in (
        "Multistate Response Behavior (MSRB)",
        "### Study binding",
        "The model predicts the response-window phenotype, not an MSRB scalar.",
        "### End-to-end evidence path",
        "#### 1. Reader response-window reduction",
        "#### 2. Within-experiment replicate handling",
        "#### 3. Study-owned repeat adjudication and label promotion",
        "#### 4. Four-state response-window phenotype",
        "[r00, r10, r01, r11, b00, b10, b01, b11]",
        "#### 5. Sequence-to-phenotype prediction",
        "#### 6. MSRB scoring",
        "Rounded value",
        "500 joint bootstrap draws",
        "#### 7. Greedy allocation and prospective measurement",
        "### Applied controls and promotion evidence",
        "### Uncertainty and censoring",
        "practical no-go outcome",
        "rectangular colorbar",
        "### Claim boundaries",
    ):
        assert required in text
    assert all(line.strip() != "-" for line in text.splitlines())
    assert "source of truth" not in text.casefold()


def test_generic_msrb_doc_is_k_state_with_bounded_assay_mapping() -> None:
    path = Path("src/dnadesign/opal/docs/plugins/objectives/multistate-response-behavior.md")
    text = path.read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    mapping_heading = "### Worked assay mapping: dual-reporter promoter screen"
    mapping_start = text.index(mapping_heading)
    mapping_end = text.index("\n### When to use this objective", mapping_start)
    mapping = " ".join(
        line.removeprefix("> ").strip() for line in text[mapping_start:mapping_end].splitlines() if line.strip()
    )
    generic_core = text[:mapping_start] + text[mapping_end:]
    generic_core_without_study_path = generic_core.replace(
        "docs/studies/stress_ethanol_cipro_growth/contexts/opal/multistate-response-behavior.md",
        "",
    )

    for required in (
        "### From a multistate phenotype to one score",
        mapping_heading,
        "For four states, `2 × 4 = 8`, so the phenotype has eight values",
        "The measured state panel determines the width; MSRB does not.",
        "`K` counts states, not experimental factors",
        "### What the same-state reference means",
        "### Why normalization scales are needed",
        "The scales balance coordinate precision, not biological importance.",
        "Basic question:",
        "The final scalar applies the same smooth bottleneck",
        "### One complete four-state example",
        "the summary can still be positive while one pair is reversed",
        "not appended to the `2K` phenotype as another component",
        "state_ids: [state_a, state_b, state_c]",
    ):
        assert required in normalized
    for required in (
        "The assay choices are not part of the objective.",
        "OD600, YFP, and CFP time series",
        "[r00, r10, r01, r11, b00, b10, b01, b11]",
        "pDual-10",
        "BBa_J23105",
        "`0.308` log2",
        "`0.313` log2",
        "not limits of detection",
    ):
        assert required in mapping
    assert "Partially ON targets, exact setpoints, ordinal preferences, and don't-care states" in normalized
    assert 'state_ids: ["00", "10", "01", "11"]' not in generic_core
    assert "| State panel | `00`: no perturbation" not in generic_core
    for study_token in (
        "pDual-10",
        "BBa_J23105",
        "4–8 hours",
        "OD600",
        "YFP",
        "CFP",
        "ethanol",
        "ciprofloxacin",
        "Reader bootstrap",
    ):
        assert study_token.casefold() not in generic_core_without_study_path.casefold()
    assert "source of truth" not in text.casefold()


def test_response_window_observation_operator_docs_do_not_route_to_moving_reader_output() -> None:
    path = STUDY_ROOT / "response_window_observations" / "README.md"
    text = path.read_text(encoding="utf-8")

    assert "response_window_observations/4_8h_v1" in text
    assert "source_manifests.reader_bundle_sha256" in text
    assert "--allowed-root" in text
    assert "stress_response_window/latest" not in text


def test_sfxi_source_evidence_root_is_study_owned() -> None:
    assert SFXI_ROUND0_SOURCE_EVIDENCE_ROOT == (STUDY_ROOT / "workbench" / "source_evidence" / "opal_sfxi_round0")


def test_rmf_comparator_is_study_owned_and_not_an_executable_campaign() -> None:
    assert RMF_ROUND0_SOURCE_EVIDENCE_ROOT == (STUDY_ROOT / "workbench" / "source_evidence" / "opal_rmf_round0")
    frozen = RMF_ROUND0_SOURCE_EVIDENCE_ROOT / "secg_rmf_greedy" / "configs" / "campaign.yaml"
    assert frozen.is_file()
    assert not (OPAL_CAMPAIGNS_ROOT / "secg_rmf_greedy").joinpath("configs", "campaign.yaml").exists()
