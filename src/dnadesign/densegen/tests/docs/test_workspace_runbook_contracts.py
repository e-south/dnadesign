"""
--------------------------------------------------------------------------------
<densegen project>
src/dnadesign/densegen/tests/docs/test_workspace_runbook_contracts.py

Contract checks for packaged DenseGen workspace runbooks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKSPACES = ROOT / "workspaces"
WORKSPACE_IDS = (
    "demo_tfbs_baseline",
    "demo_sampling_baseline",
    "study_constitutive_sigma_panel",
    "study_stress_ethanol_cipro",
)
USR_WORKSPACE_IDS = (
    "demo_sampling_baseline",
    "study_constitutive_sigma_panel",
    "study_stress_ethanol_cipro",
)


def _assert_token_order(text: str, tokens: list[str], *, label: str) -> None:
    cursor = -1
    for token in tokens:
        idx = text.find(token, cursor + 1)
        assert idx >= 0, f"{label}: missing token: {token}"
        assert idx > cursor, f"{label}: out-of-order token: {token}"
        cursor = idx


def _read_runbook(workspace_id: str) -> str:
    runbook = WORKSPACES / workspace_id / "runbook.md"
    assert runbook.exists(), f"Missing runbook for workspace '{workspace_id}': {runbook}"
    return runbook.read_text()


def _extract_single_command(text: str) -> str:
    marker = "Run this single command to do everything below:"
    idx = text.find(marker)
    assert idx >= 0, "missing single-command marker"
    for raw in text[idx + len(marker) :].splitlines():
        if raw.startswith("    "):
            value = raw.strip()
            if value:
                return value
    raise AssertionError("missing single-command line")


def _extract_step_commands(text: str) -> list[str]:
    marker = "### Step-by-Step Commands"
    idx = text.find(marker)
    assert idx >= 0, "missing Step-by-Step Commands section"
    commands: list[str] = []
    in_code = False
    for raw in text[idx + len(marker) :].splitlines():
        if raw.startswith("    "):
            in_code = True
            line = raw[4:].strip()
            if line.startswith(("uv run", "pixi run", "cruncher ")):
                commands.append(line)
            continue
        if in_code:
            if not raw.strip():
                continue
            break
    assert commands, "step-by-step block did not include executable commands"
    return commands


def test_packaged_workspace_runbooks_exist() -> None:
    expected = {
        "demo_tfbs_baseline",
        "demo_sampling_baseline",
        "study_constitutive_sigma_panel",
        "study_stress_ethanol_cipro",
    }
    found = {path.parent.name for path in WORKSPACES.glob("*/runbook.md")}
    assert expected.issubset(found), f"Missing runbooks: {sorted(expected - found)}"


def test_workspace_runbooks_follow_standard_sections() -> None:
    for workspace_id in WORKSPACE_IDS:
        text = _read_runbook(workspace_id)
        assert f"## {workspace_id} Runbook" in text
        assert "**Workspace Path**" in text
        assert "**Regulators**" in text
        assert "**Purpose**" in text
        assert "**Run This Single Command**" in text
        assert "Run this single command to do everything below:" in text
        assert "### Step-by-Step Commands" in text


def test_workspace_runbooks_requiring_cruncher_export_include_handoff_step() -> None:
    for workspace_id in USR_WORKSPACE_IDS:
        text = _read_runbook(workspace_id)
        assert "cruncher catalog export-densegen" in text
        assert "/src/dnadesign/cruncher/workspaces/" in text
        assert "/configs/config.yaml" in text


def test_workspace_runbooks_single_command_matches_canonical_step_sequence() -> None:
    canonical_tokens_by_workspace = {
        "demo_tfbs_baseline": [
            "dense validate-config --probe-solver",
            "dense run --fresh --no-plot",
            "dense inspect run --events --library",
            "dense plot",
            "dense notebook generate",
        ],
        "demo_sampling_baseline": [
            "dense workspace init",
            "--output-mode both",
            "fimo --version",
            "dense validate-config --probe-solver",
            "dense run --fresh --no-plot",
            "dense inspect run --events --library",
            "dense plot",
            "dense notebook generate",
        ],
        "study_constitutive_sigma_panel": [
            "dense workspace init",
            "--output-mode both",
            "fimo --version",
            "dense validate-config --probe-solver",
            "dense run --fresh --no-plot",
            "dense inspect run --events --library",
            "dense plot",
            "dense notebook generate",
        ],
        "study_stress_ethanol_cipro": [
            "dense workspace init",
            "--output-mode both",
            "fimo --version",
            "dense validate-config --probe-solver",
            "dense run --fresh --no-plot",
            "dense inspect run --events --library",
            "dense plot",
            "dense notebook generate",
        ],
    }
    for workspace_id in WORKSPACE_IDS:
        text = _read_runbook(workspace_id)
        single_command = _extract_single_command(text)
        step_commands_text = "\n".join(_extract_step_commands(text))
        tokens = canonical_tokens_by_workspace[workspace_id]
        _assert_token_order(single_command, list(tokens), label=f"{workspace_id}: single command")
        _assert_token_order(step_commands_text, list(tokens), label=f"{workspace_id}: step-by-step block")


def test_workspace_runbooks_keep_optional_commands_outside_canonical_step_block() -> None:
    for workspace_id in WORKSPACE_IDS:
        text = _read_runbook(workspace_id)
        step_block = "\n".join(_extract_step_commands(text))
        assert "campaign-reset" not in step_block
        assert "### Optional workspace reset" in text
        if workspace_id in USR_WORKSPACE_IDS:
            assert "cruncher catalog export-densegen" not in step_block
            assert "### Optional artifact refresh from Cruncher" in text
