"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/architecture/test_runtime_command_groups.py

Runtime command-group contract tests for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import yaml

from dnadesign.studies.units.eco1_rt_repack.tests._helpers import repo_root

_PIPELINE = "docs/studies/eco1_rt_repack/operations/runtime/command-groups/pipeline.yaml"
_README = "docs/studies/eco1_rt_repack/operations/runtime/command-groups/README.md"

_EXECUTABLE_LANES = {
    "structure_authority",
    "structure_preprocessing",
    "contact_profile",
    "contact_geometry_profile",
    "conservation_provider_sources",
    "conservation_roster_cache",
    "conservation_source_bundles",
    "conservation_source_sufficiency",
    "conservation_alignments",
    "conservation_visualizations",
    "evidence_profiles",
    "manual_mask_authority",
    "mask_contract",
    "contact_risk_profile",
    "sampling_plan",
    "sample_ingest",
    "candidate_table",
    "foldcheck_request",
    "phase0_contract_validation",
    "phase1_contract_validation",
    "phase2_contract_validation",
}
_EXTERNAL_LANES = {"colabfold_scc_smoke"}
_PLANNED_LANES = {
    "refine_dev_spec",
    "foldcheck_report",
    "assembly_feasibility",
    "candidate_handoff",
    "rt_lnrna_handoff",
}


def _pipeline() -> dict[str, object]:
    with (repo_root() / _PIPELINE).open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    assert isinstance(loaded, dict)
    return loaded


def test_runtime_command_group_readme_is_no_longer_placeholder() -> None:
    text = (repo_root() / _README).read_text(encoding="utf-8")

    assert "placeholder route" not in text
    assert "not a hidden run-all pipeline" in text
    assert "simple clade9-plurality-25/direct-contact-5 A" in text
    assert "### Rerun Ladder" in text
    assert "### Source-Role Guardrails" in text
    assert "Tao is the masking-method prior" in text


def test_pipeline_names_sequential_executable_lanes() -> None:
    pipeline = _pipeline()
    lanes = pipeline["lanes"]
    assert isinstance(lanes, list)
    by_id = {lane["id"]: lane for lane in lanes}

    assert _EXECUTABLE_LANES.issubset(by_id)
    for lane_id in sorted(_EXECUTABLE_LANES):
        command = by_id[lane_id].get("command")
        assert isinstance(command, dict), lane_id
        argv = command.get("argv")
        assert isinstance(argv, list), lane_id
        assert "python" in argv or lane_id == "conservation_alignments"

    for lane_id in sorted(_EXTERNAL_LANES):
        command = by_id[lane_id].get("command")
        assert isinstance(command, dict), lane_id
        assert command.get("argv", [])[0] == "qsub"

    for lane_id in sorted(_PLANNED_LANES):
        assert by_id[lane_id].get("command") is None, lane_id


def test_pipeline_preserves_study_aligner_thread_boundaries() -> None:
    by_id = {lane["id"]: lane for lane in _pipeline()["lanes"]}

    assert by_id["conservation_alignments"]["owner"] == "eco1_rt_repack"
    assert by_id["conservation_visualizations"]["owner"] == "aligner.msa"
    assert by_id["sampling_plan"]["owner"] == "eco1_rt_repack"
    assert by_id["sample_ingest"]["owner"] == "thread"
    assert by_id["foldcheck_request"]["owner"] == "eco1_rt_repack"
    assert by_id["colabfold_scc_smoke"]["owner"] == "bu_scc_runtime"
    assert by_id["foldcheck_report"]["owner"] == "planned_thread"
    assert by_id["mask_contract"]["owner"] == "eco1_rt_repack"
    assert by_id["contact_risk_profile"]["owner"] == "eco1_rt_repack"
    assert "pixi" in by_id["conservation_alignments"]["command"]["argv"]
    assert "dnadesign.aligner.msa.visualization" in by_id["conservation_visualizations"]["command"]["argv"]
