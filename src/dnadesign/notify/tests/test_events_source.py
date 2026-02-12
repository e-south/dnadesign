"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/tests/test_events_source.py

Tool events-source resolver tests for notify setup workflows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from dnadesign.notify.errors import NotifyConfigError
from dnadesign.notify.events_source import register_tool_events_source, resolve_tool_events_path


def test_resolve_tool_events_path_infer_evo2_from_single_usr_writeback_job(tmp_path: Path) -> None:
    config = tmp_path / "infer.yaml"
    usr_root = tmp_path / "usr_root"
    config.write_text(
        "\n".join(
            [
                "model:",
                "  id: evo2",
                "  device: cpu",
                "  precision: fp32",
                "  alphabet: dna",
                "jobs:",
                "  - id: j1",
                "    operation: generate",
                "    ingest:",
                "      source: usr",
                "      dataset: infer_demo",
                f"      root: {usr_root}",
                "    params:",
                "      max_new_tokens: 8",
                "    io:",
                "      write_back: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    events_path, policy = resolve_tool_events_path(tool="infer_evo2", config=config)

    assert events_path == (usr_root / "infer_demo" / ".events.log").resolve()
    assert policy == "infer_evo2"


def test_resolve_tool_events_path_infer_evo2_uses_env_usr_root_when_ingest_root_absent(
    tmp_path: Path, monkeypatch
) -> None:
    config = tmp_path / "infer.yaml"
    usr_root = tmp_path / "env_usr_root"
    config.write_text(
        "\n".join(
            [
                "model:",
                "  id: evo2",
                "  device: cpu",
                "  precision: fp32",
                "  alphabet: dna",
                "jobs:",
                "  - id: j1",
                "    operation: generate",
                "    ingest:",
                "      source: usr",
                "      dataset: infer_demo",
                "    params:",
                "      max_new_tokens: 8",
                "    io:",
                "      write_back: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DNADESIGN_USR_ROOT", str(usr_root))

    events_path, policy = resolve_tool_events_path(tool="infer_evo2", config=config)

    assert events_path == (usr_root / "infer_demo" / ".events.log").resolve()
    assert policy == "infer_evo2"


def test_resolve_tool_events_path_infer_evo2_requires_explicit_root_without_env(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "infer.yaml"
    config.write_text(
        "\n".join(
            [
                "model:",
                "  id: evo2",
                "  device: cpu",
                "  precision: fp32",
                "  alphabet: dna",
                "jobs:",
                "  - id: j1",
                "    operation: generate",
                "    ingest:",
                "      source: usr",
                "      dataset: infer_demo",
                "    params:",
                "      max_new_tokens: 8",
                "    io:",
                "      write_back: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("DNADESIGN_USR_ROOT", raising=False)

    with pytest.raises(NotifyConfigError, match="requires ingest.root or DNADESIGN_USR_ROOT"):
        resolve_tool_events_path(tool="infer_evo2", config=config)


def test_resolve_tool_events_path_infer_evo2_rejects_ambiguous_destinations(tmp_path: Path) -> None:
    config = tmp_path / "infer.yaml"
    config.write_text(
        "\n".join(
            [
                "model:",
                "  id: evo2",
                "  device: cpu",
                "  precision: fp32",
                "  alphabet: dna",
                "jobs:",
                "  - id: j1",
                "    operation: generate",
                "    ingest:",
                "      source: usr",
                "      dataset: ds_a",
                "      root: /tmp/usr_a",
                "    params:",
                "      max_new_tokens: 8",
                "    io:",
                "      write_back: true",
                "  - id: j2",
                "    operation: generate",
                "    ingest:",
                "      source: usr",
                "      dataset: ds_b",
                "      root: /tmp/usr_b",
                "    params:",
                "      max_new_tokens: 8",
                "    io:",
                "      write_back: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(NotifyConfigError, match="multiple USR destinations"):
        resolve_tool_events_path(tool="infer_evo2", config=config)


def test_resolve_tool_events_path_infer_evo2_requires_usr_writeback_job(tmp_path: Path) -> None:
    config = tmp_path / "infer.yaml"
    config.write_text(
        "\n".join(
            [
                "model:",
                "  id: evo2",
                "  device: cpu",
                "  precision: fp32",
                "  alphabet: dna",
                "jobs:",
                "  - id: j1",
                "    operation: generate",
                "    ingest:",
                "      source: sequences",
                "    params:",
                "      max_new_tokens: 8",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(NotifyConfigError, match="ingest.source='usr'"):
        resolve_tool_events_path(tool="infer_evo2", config=config)


def test_register_tool_events_source_supports_custom_tool(tmp_path: Path) -> None:
    config = tmp_path / "custom.yaml"
    config.write_text("x: 1\n", encoding="utf-8")
    resolved = tmp_path / "custom" / ".events.log"

    register_tool_events_source(
        tool="custom_tool",
        resolver=lambda path: resolved if path == config else Path("unexpected"),
        default_policy="custom_tool",
        aliases=("custom-tool",),
    )

    events_path, policy = resolve_tool_events_path(tool="custom-tool", config=config)
    assert events_path == resolved
    assert policy == "custom_tool"


def test_register_tool_events_source_rejects_duplicate_alias() -> None:
    register_tool_events_source(
        tool="custom_alpha",
        resolver=lambda path: path,
        aliases=("custom-alpha",),
    )
    with pytest.raises(NotifyConfigError, match="alias 'custom-alpha' is already registered"):
        register_tool_events_source(
            tool="custom_beta",
            resolver=lambda path: path,
            aliases=("custom-alpha",),
        )


def test_events_source_module_is_registry_only() -> None:
    import dnadesign.notify.events_source as events_source_module

    parsed = ast.parse(inspect.getsource(events_source_module))
    imported_modules: set[str] = set()
    for node in ast.walk(parsed):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            imported_modules.add(str(node.module or ""))

    assert "yaml" not in imported_modules
    assert "os" not in imported_modules
