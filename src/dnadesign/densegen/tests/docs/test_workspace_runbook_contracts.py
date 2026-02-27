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


def _read_runbook_script(workspace_id: str) -> str:
    script = WORKSPACES / workspace_id / "runbook.sh"
    assert script.exists(), f"Missing runbook script for workspace '{workspace_id}': {script}"
    return script.read_text()


def _read_shared_runbook_lib() -> str:
    lib_path = WORKSPACES / "_shared" / "runbook_lib.sh"
    assert lib_path.exists(), f"Missing shared runbook library: {lib_path}"
    return lib_path.read_text()


def _extract_single_command(text: str) -> str:
    marker = "Run this command from the workspace root:"
    idx = text.find(marker)
    assert idx >= 0, "missing runbook-command marker"
    for raw in text[idx + len(marker) :].splitlines():
        if raw.startswith("    "):
            value = raw.strip()
            if value and not value.startswith("#"):
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
        assert "**Runbook command**" in text
        assert "Run this command from the workspace root:" in text
        assert "### Step-by-Step Commands" in text


def test_workspace_runbooks_requiring_cruncher_export_include_handoff_step() -> None:
    for workspace_id in USR_WORKSPACE_IDS:
        text = _read_runbook(workspace_id)
        assert "cruncher catalog export-densegen" in text
        assert "/src/dnadesign/cruncher/workspaces/" in text
        assert "/configs/config.yaml" in text


def test_workspace_runbooks_include_workspace_local_runbook_script() -> None:
    expected_runner = {
        "demo_tfbs_baseline": "uv",
        "demo_sampling_baseline": "pixi",
        "study_constitutive_sigma_panel": "pixi",
        "study_stress_ethanol_cipro": "pixi",
    }
    expected_usr_registry = {
        "demo_tfbs_baseline": "false",
        "demo_sampling_baseline": "true",
        "study_constitutive_sigma_panel": "true",
        "study_stress_ethanol_cipro": "true",
    }
    expected_require_fimo = {
        "demo_tfbs_baseline": "false",
        "demo_sampling_baseline": "true",
        "study_constitutive_sigma_panel": "true",
        "study_stress_ethanol_cipro": "true",
    }
    for workspace_id in WORKSPACE_IDS:
        text = _read_runbook(workspace_id)
        assert "./runbook.sh" in text
        assert "REPO_ROOT=" not in text
        script = _read_runbook_script(workspace_id)
        assert script.startswith("#!/usr/bin/env bash\n")
        _assert_token_order(
            script,
            [
                "set -euo pipefail",
                'CONFIG="$PWD/config.yaml"',
                'NOTEBOOK="$PWD/outputs/notebooks/densegen_run_overview.py"',
                'source "$SCRIPT_DIR/../_shared/runbook_lib.sh"',
                "densegen_runbook_main",
                '--config "$CONFIG"',
                '--notebook "$NOTEBOOK"',
                f'--runner "{expected_runner[workspace_id]}"',
                f'--ensure-usr-registry "{expected_usr_registry[workspace_id]}"',
                f'--require-fimo "{expected_require_fimo[workspace_id]}"',
            ],
            label=f"{workspace_id}: runbook.sh",
        )


def test_usr_workspace_runbooks_seed_registry_before_dense_run() -> None:
    for workspace_id in USR_WORKSPACE_IDS:
        script = _read_runbook_script(workspace_id)
        assert '--ensure-usr-registry "true"' in script

        runbook_text = _read_runbook(workspace_id)
        assert 'USR_REGISTRY="$PWD/outputs/usr_datasets/registry.yaml"' in runbook_text
        assert (
            'ROOT_REGISTRY="$(git rev-parse --show-toplevel)/src/dnadesign/usr/datasets/registry.yaml"' in runbook_text
        )


def test_shared_runbook_lib_keeps_canonical_dense_command_sequence() -> None:
    script = _read_shared_runbook_lib()
    _assert_token_order(
        script,
        [
            "_densegen_require_command uv",
            "_densegen_require_command git",
            'if [[ "$runner" == "pixi" || "$require_fimo" == "true" ]]; then',
            'if [[ "$ensure_usr_registry" == "true" ]]; then',
            '"${dense_cmd[@]}" validate-config --probe-solver -c "$config"',
            '"${dense_cmd[@]}" run --fresh --no-plot -c "$config"',
            '"${dense_cmd[@]}" inspect run --events --library -c "$config"',
            '"${dense_cmd[@]}" plot -c "$config"',
            '"${dense_cmd[@]}" notebook generate -c "$config"',
        ],
        label="workspaces/_shared/runbook_lib.sh canonical sequence",
    )


def test_shared_runbook_lib_enforces_config_and_notebook_artifact_guards() -> None:
    script = _read_shared_runbook_lib()
    _assert_token_order(
        script,
        [
            'if [[ ! -f "$config" ]]; then',
            'echo "DenseGen config not found at: $config" >&2',
            'if [[ "$ensure_usr_registry" == "true" ]]; then',
            'if [[ ! -f "$root_registry" ]]; then',
            'echo "USR registry source not found at: $root_registry" >&2',
            '"${dense_cmd[@]}" notebook generate -c "$config"',
            'if [[ ! -f "$notebook" ]]; then',
            'echo "DenseGen notebook was not generated at: $notebook" >&2',
            'uv run marimo check "$notebook"',
        ],
        label="workspaces/_shared/runbook_lib.sh guardrails",
    )


def test_workspace_runbooks_single_command_matches_step_sequence() -> None:
    single_command_tokens_by_workspace = {
        "demo_tfbs_baseline": [
            "./runbook.sh",
        ],
        "demo_sampling_baseline": [
            "./runbook.sh",
        ],
        "study_constitutive_sigma_panel": [
            "./runbook.sh",
        ],
        "study_stress_ethanol_cipro": [
            "./runbook.sh",
        ],
    }
    step_tokens_by_workspace = {
        "demo_tfbs_baseline": [
            "dense validate-config --probe-solver",
            "dense run --fresh --no-plot",
            "dense inspect run --events --library",
            "dense plot",
            "dense notebook generate",
        ],
        "demo_sampling_baseline": [
            "fimo --version",
            "dense validate-config --probe-solver",
            "dense run --fresh --no-plot",
            "dense inspect run --events --library",
            "dense plot",
            "dense notebook generate",
        ],
        "study_constitutive_sigma_panel": [
            "fimo --version",
            "dense validate-config --probe-solver",
            "dense run --fresh --no-plot",
            "dense inspect run --events --library",
            "dense plot",
            "dense notebook generate",
        ],
        "study_stress_ethanol_cipro": [
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
        _assert_token_order(
            single_command,
            list(single_command_tokens_by_workspace[workspace_id]),
            label=f"{workspace_id}: single command",
        )
        _assert_token_order(
            step_commands_text,
            list(step_tokens_by_workspace[workspace_id]),
            label=f"{workspace_id}: step-by-step block",
        )


def test_workspace_runbooks_keep_optional_commands_outside_canonical_step_block() -> None:
    for workspace_id in WORKSPACE_IDS:
        text = _read_runbook(workspace_id)
        step_block = "\n".join(_extract_step_commands(text))
        assert "campaign-reset" not in step_block
        assert "### Optional workspace reset" in text
        if workspace_id in USR_WORKSPACE_IDS:
            assert "cruncher catalog export-densegen" not in step_block
            assert "### Optional artifact refresh from Cruncher" in text
