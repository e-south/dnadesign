"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/docs/test_workspace_runbook_contracts.py

Validate that each workspace runbook encodes the canonical source-merging
discovery lifecycle used for optimization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[2] / "workspaces"


def _load_config(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text())
    assert isinstance(payload, dict)
    cruncher_payload = payload.get("cruncher")
    assert isinstance(cruncher_payload, dict)
    return cruncher_payload


def _load_machine_runbook(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text())
    assert isinstance(payload, dict)
    runbook = payload.get("runbook")
    assert isinstance(runbook, dict)
    return runbook


def _assert_token_order(text: str, tokens: list[str], *, label: str) -> None:
    cursor = -1
    for token in tokens:
        idx = text.find(token, cursor + 1)
        assert idx >= 0, f"{label}: missing token: {token}"
        assert idx > cursor, f"{label}: out-of-order token: {token}"
        cursor = idx


def _machine_runbook_commands(path: Path) -> list[str]:
    runbook = _load_machine_runbook(path)
    steps = runbook.get("steps")
    assert isinstance(steps, list) and steps
    commands: list[str] = []
    for item in steps:
        assert isinstance(item, dict)
        run = item.get("run")
        assert isinstance(run, list) and run
        command = "cruncher " + " ".join(str(token) for token in run)
        commands.append(command)
    return commands


def _extract_step_by_step_commands(text: str) -> list[str]:
    marker = "### Step-by-Step Commands"
    idx = text.find(marker)
    assert idx >= 0, "missing step-by-step section"
    lines = text[idx + len(marker) :].splitlines()
    in_code = False
    commands: list[str] = []
    for raw in lines:
        if raw.startswith("    "):
            in_code = True
            line = raw[4:].strip()
            if line.startswith("cruncher "):
                commands.append(line)
            continue
        if in_code:
            if not raw.strip():
                continue
            break
    assert commands, "step-by-step section did not include cruncher commands"
    return commands


def _normalize_command(value: str) -> str:
    text = str(value).strip()
    text = text.replace('-c "$CONFIG"', "-c configs/config.yaml")
    return " ".join(text.split())


def test_every_workspace_with_config_has_sibling_runbook() -> None:
    root = _workspace_root()
    for config_path in sorted(root.glob("*/configs/config.yaml")):
        runbook_path = config_path.parent.parent / "runbook.md"
        assert runbook_path.exists(), f"missing runbook for workspace: {config_path.parent.parent.name}"


def test_workspace_runbooks_start_with_single_command_then_step_by_step_block() -> None:
    root = _workspace_root()
    for config_path in sorted(root.glob("*/configs/config.yaml")):
        workspace = config_path.parent.parent
        runbook_path = workspace / "runbook.md"
        text = runbook_path.read_text()
        assert text.startswith("## "), f"{workspace.name}: runbook top header must use ##"
        assert "**Workspace Path**" in text, f"{workspace.name}: runbook missing bold Workspace Path label"
        assert "**Regulators**" in text, f"{workspace.name}: runbook missing bold Regulators label"
        assert "**Purpose**" in text, f"{workspace.name}: runbook missing bold Purpose label"
        assert "**Run This Single Command**" in text, (
            f"{workspace.name}: runbook missing bold Run This Single Command label"
        )
        assert "### Step-by-Step Commands" in text, (
            f"{workspace.name}: runbook missing ### Step-by-Step Commands section"
        )
        assert "### Workspace Path" not in text, f"{workspace.name}: Workspace Path should be bold label, not subheader"
        assert "### Regulators" not in text, f"{workspace.name}: Regulators should be bold label, not subheader"
        assert "### Purpose" not in text, f"{workspace.name}: Purpose should be bold label, not subheader"
        assert "### Run This Single Command" not in text, (
            f"{workspace.name}: Run This Single Command should be bold label, not subheader"
        )
        assert "Run this single command to do everything below:" in text, (
            f"{workspace.name}: runbook should present a one-command end-to-end path near the top"
        )
        assert "uv run cruncher workspaces run --runbook configs/runbook.yaml" in text, (
            f"{workspace.name}: runbook missing one-command end-to-end invocation"
        )
        _assert_token_order(
            text,
            [
                "**Workspace Path**",
                "**Regulators**",
                "**Purpose**",
                "**Run This Single Command**",
                "Run this single command to do everything below:",
                "uv run cruncher workspaces run --runbook configs/runbook.yaml",
                "### Step-by-Step Commands",
            ],
            label=f"{workspace.name}/runbook.md",
        )


def test_every_workspace_with_config_has_machine_runbook() -> None:
    root = _workspace_root()
    for config_path in sorted(root.glob("*/configs/config.yaml")):
        runbook_yaml = config_path.parent / "runbook.yaml"
        assert runbook_yaml.exists(), f"missing configs/runbook.yaml for workspace: {config_path.parent.parent.name}"


def test_machine_runbooks_include_fail_fast_analysis_and_export_steps() -> None:
    root = _workspace_root()
    for config_path in sorted(root.glob("*/configs/config.yaml")):
        workspace = config_path.parent.parent
        runbook_yaml = workspace / "configs" / "runbook.yaml"
        runbook = _load_machine_runbook(runbook_yaml)
        steps = runbook.get("steps")
        assert isinstance(steps, list) and steps
        ids = [str(item.get("id")) for item in steps if isinstance(item, dict)]
        assert len(ids) == len(set(ids)), f"{workspace.name}: duplicate machine runbook step ids"
        assert "reset_workspace" in ids, f"{workspace.name}: missing reset_workspace step"
        assert "clean_transient" not in ids, f"{workspace.name}: replace clean_transient with reset_workspace"
        assert "analyze_summary" in ids, f"{workspace.name}: missing analyze_summary step"
        assert ("export_sequences_latest" in ids) or ("export_sequences_outputs" in ids), (
            f"{workspace.name}: missing export sequence step for handoff readiness"
        )


def test_machine_runbook_study_steps_include_human_readable_descriptions() -> None:
    root = _workspace_root()
    for config_path in sorted(root.glob("*/configs/config.yaml")):
        workspace = config_path.parent.parent
        runbook_yaml = workspace / "configs" / "runbook.yaml"
        runbook = _load_machine_runbook(runbook_yaml)
        steps = runbook.get("steps")
        assert isinstance(steps, list) and steps
        by_id = {str(item.get("id")): item for item in steps if isinstance(item, dict)}
        for step_id in ("study_run_length_vs_score", "study_run_diversity_vs_score"):
            step = by_id.get(step_id)
            assert isinstance(step, dict), f"{workspace.name}: missing {step_id} step"
            description = step.get("description")
            assert isinstance(description, str) and description.strip(), (
                f"{workspace.name}: {step_id} should include a non-empty description for runbook readability"
            )


def test_step_by_step_commands_are_coupled_to_machine_runbooks() -> None:
    root = _workspace_root()
    for runbook_yaml in sorted(root.glob("*/configs/runbook.yaml")):
        workspace = runbook_yaml.parent.parent
        runbook_md = workspace / "runbook.md"
        text = runbook_md.read_text()
        expected = [_normalize_command(item) for item in _machine_runbook_commands(runbook_yaml)]
        observed = [_normalize_command(item) for item in _extract_step_by_step_commands(text)]
        assert observed == expected, (
            f"{workspace.name}: step-by-step command sequence must exactly mirror configs/runbook.yaml"
        )


def test_workspace_runbooks_encode_source_merge_then_meme_oops_flow() -> None:
    root = _workspace_root()
    for config_path in sorted(root.glob("*/configs/config.yaml")):
        workspace = config_path.parent.parent
        runbook_path = workspace / "runbook.md"
        runbook = runbook_path.read_text()
        cfg = _load_config(config_path)

        workspace_payload = cfg.get("workspace")
        assert isinstance(workspace_payload, dict)
        regulators = workspace_payload.get("regulator_sets")
        assert isinstance(regulators, list) and regulators
        first_set = regulators[0]
        assert isinstance(first_set, list) and first_set
        regulator_names = [str(tf) for tf in first_set]

        discover = cfg.get("discover")
        assert isinstance(discover, dict)
        discover_source_id = discover.get("source_id")
        assert isinstance(discover_source_id, str) and discover_source_id

        _assert_token_order(
            runbook,
            [
                "fetch sites --source",
                "fetch sites --source regulondb",
                "discover motifs",
                "--tool meme --meme-mod oops",
                f"--source-id {discover_source_id}",
                'lock -c "$CONFIG"',
                'parse --force-overwrite -c "$CONFIG"',
                'sample --force-overwrite -c "$CONFIG"',
                'analyze --summary -c "$CONFIG"',
                'export sequences --latest -c "$CONFIG"',
            ],
            label=f"{workspace.name}/runbook.md",
        )

        assert "merges all fetched site sets across sources" in runbook
        assert "fetch sites --source regulondb" in runbook
        for tf_name in regulator_names:
            assert f"--tf {tf_name}" in runbook, f"{workspace.name}: missing TF flag for {tf_name}"

        local_input = workspace / "inputs" / "local_motifs"
        if local_input.is_dir():
            local_tfs = sorted(path.stem for path in local_input.glob("*.txt"))
            if local_tfs:
                assert "fetch sites --source demo_local_meme" in runbook
                for tf_name in set(local_tfs) & set(regulator_names):
                    assert f"--tf {tf_name}" in runbook, f"{workspace.name}: missing local TF {tf_name}"

        if "baeR" in regulator_names:
            assert "fetch sites --source baer_chip_exo --tf baeR" in runbook


def test_workspace_configs_keep_discovery_source_contracts() -> None:
    root = _workspace_root()
    for config_path in sorted(root.glob("*/configs/config.yaml")):
        workspace = config_path.parent.parent
        cfg = _load_config(config_path)

        workspace_payload = cfg.get("workspace")
        assert isinstance(workspace_payload, dict)
        regulators = workspace_payload.get("regulator_sets")
        assert isinstance(regulators, list) and regulators
        first_set = regulators[0]
        assert isinstance(first_set, list) and first_set
        regulator_names = {str(tf) for tf in first_set}

        discover = cfg.get("discover")
        assert isinstance(discover, dict)
        discover_source_id = discover.get("source_id")
        assert isinstance(discover_source_id, str) and discover_source_id
        assert discover.get("tool") == "meme", f"{workspace.name}: discovery tool must be meme"
        assert discover.get("meme_mod") == "oops", f"{workspace.name}: discovery meme_mod must be oops"

        catalog = cfg.get("catalog")
        assert isinstance(catalog, dict)
        assert catalog.get("source_preference") == [discover_source_id], (
            f"{workspace.name}: catalog.source_preference must point to discovered source_id"
        )

        ingest = cfg.get("ingest")
        assert isinstance(ingest, dict)
        regulondb = ingest.get("regulondb")
        assert isinstance(regulondb, dict)
        assert regulondb.get("curated_sites") is True, f"{workspace.name}: regulondb curated_sites must be true"
        assert regulondb.get("ht_sites") is False, f"{workspace.name}: regulondb ht_sites must be false"

        local_sources = ingest.get("local_sources")
        assert isinstance(local_sources, list)
        source_ids = {entry.get("source_id") for entry in local_sources if isinstance(entry, dict)}
        local_input = workspace / "inputs" / "local_motifs"
        if local_input.is_dir() and list(local_input.glob("*.txt")):
            assert "demo_local_meme" in source_ids, f"{workspace.name}: missing demo_local_meme local source"

        site_sources = ingest.get("site_sources")
        if site_sources is None:
            site_sources = []
        assert isinstance(site_sources, list)
        site_source_ids = {entry.get("source_id") for entry in site_sources if isinstance(entry, dict)}
        if "baeR" in regulator_names:
            assert "baer_chip_exo" in site_source_ids, f"{workspace.name}: missing baer_chip_exo site source"
