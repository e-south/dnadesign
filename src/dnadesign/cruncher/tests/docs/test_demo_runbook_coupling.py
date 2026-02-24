"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/docs/test_demo_runbook_coupling.py

Couple demo doc command flows to workspace machine runbook contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _normalize_command(value: str) -> str:
    text = str(value).strip()
    text = text.replace('-c "$CONFIG"', "-c configs/config.yaml")
    return " ".join(text.split())


def _machine_runbook_commands(workspace_name: str) -> list[str]:
    path = ROOT / "workspaces" / workspace_name / "configs" / "runbook.yaml"
    payload = yaml.safe_load(path.read_text())
    runbook = payload.get("runbook")
    assert isinstance(runbook, dict)
    steps = runbook.get("steps")
    assert isinstance(steps, list) and steps
    commands: list[str] = []
    for item in steps:
        assert isinstance(item, dict)
        run = item.get("run")
        assert isinstance(run, list) and run
        commands.append(_normalize_command("cruncher " + " ".join(str(token) for token in run)))
    return commands


def _fenced_block_after_heading(text: str, heading: str) -> str:
    idx = text.find(heading)
    assert idx >= 0, f"missing heading: {heading}"
    lines = text[idx + len(heading) :].splitlines()
    in_fence = False
    block: list[str] = []
    for raw in lines:
        if raw.strip().startswith("```"):
            if in_fence:
                break
            in_fence = True
            continue
        if in_fence:
            block.append(raw)
    assert block, f"missing fenced code block after heading: {heading}"
    return "\n".join(block)


def _doc_commands_from_block(block: str) -> list[str]:
    commands: list[str] = []
    for raw in block.splitlines():
        line = raw.strip()
        if line.startswith("uv run cruncher "):
            commands.append(_normalize_command("cruncher " + line.removeprefix("uv run cruncher ")))
        elif line.startswith("cruncher "):
            commands.append(_normalize_command(line))
    assert commands, "demo block did not include cruncher commands"
    return commands


def test_demo_docs_include_one_command_machine_runbook_entrypoint() -> None:
    cases = [
        ROOT / "docs" / "demos" / "demo_pairwise.md",
        ROOT / "docs" / "demos" / "demo_multitf.md",
    ]
    for doc_path in cases:
        text = doc_path.read_text()
        assert "uv run cruncher workspaces run --runbook configs/runbook.yaml" in text, (
            f"{doc_path.name}: must include the one-command machine runbook entrypoint"
        )


def test_project_all_tfs_demo_command_block_matches_workspace_machine_runbook_order() -> None:
    doc_path = ROOT / "docs" / "demos" / "project_all_tfs.md"
    workspace_name = "project_tfs_lexa_cpxr_baer_rcda_lrp_fur_fnr_acrr_soxr_soxs"
    text = doc_path.read_text()
    block = _fenced_block_after_heading(text, "### End-to-end commands")
    observed = _doc_commands_from_block(block)
    expected = _machine_runbook_commands(workspace_name)
    assert observed[: len(expected)] == expected, (
        f"{doc_path.name}: command flow must start with workspace machine runbook order from {workspace_name}"
    )
