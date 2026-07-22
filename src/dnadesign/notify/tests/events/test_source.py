"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/tests/events/test_source.py

Tool events-source resolver tests for notify setup workflows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from dnadesign.notify.core.errors import NotifyConfigError
from dnadesign.notify.events.source import register_tool_events_source, resolve_tool_events_path


def _write_construct_config(
    config_path: Path,
    *,
    input_root: str | Path | None,
    output_root: str | Path | None,
    output_dataset: str | None = "construct/demo_window",
) -> None:
    lines = [
        "job:",
        "  id: slot_a_window",
        "  input:",
        "    source:",
        "      kind: usr",
        "      dataset: anchors_demo",
    ]
    if input_root is not None:
        lines.append(f"      root: {Path(input_root).as_posix() if isinstance(input_root, Path) else input_root}")
    lines.extend(
        [
            "  template:",
            "    id: template_demo",
            "    source:",
            "      kind: literal",
            "      sequence: AAAATTTTCCCCGGGG",
            "    circular: true",
            "  parts:",
            "    - name: anchor",
            "      role: anchor",
            "      sequence:",
            "        source: input_field",
            "        field: sequence",
            "      placement:",
            "        kind: replace",
            "        orientation: forward",
            "        locator:",
            "          kind: coordinates",
            "          start: 8",
            "          end: 12",
            "        guards:",
            "          replaced_sequence: CCCC",
            "  realize:",
            "    mode: window",
            "    focal_part: anchor",
            "    window:",
            "      semantics: fixed_total",
            "      reference: center",
            "      direction: symmetric",
            "      size_bp: 8",
            "      offset_bp: 0",
            "  output:",
            "    target:",
            "      kind: usr",
        ]
    )
    if output_dataset is not None:
        lines.append(f"      dataset: {output_dataset}")
    if output_root is not None:
        lines.append(f"      root: {Path(output_root).as_posix() if isinstance(output_root, Path) else output_root}")
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_resolve_tool_events_path_infer_from_single_usr_writeback_job(tmp_path: Path) -> None:
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

    events_path, policy = resolve_tool_events_path(tool="infer", config=config)

    assert events_path == (usr_root / "infer_demo" / ".events.log").resolve()
    assert policy == "infer"


def test_resolve_tool_events_path_infer_from_single_sequence_view_dataset(tmp_path: Path) -> None:
    config = tmp_path / "infer_sequence_views.yaml"
    usr_root = tmp_path / "usr_root"
    config.write_text(
        "\n".join(
            [
                "model:",
                "  id: evo2_7b",
                "  device: cpu",
                "  precision: fp32",
                "  alphabet: dna",
                "jobs:",
                "  - id: anchor_sequence_views_7b",
                "    operation: extract",
                "    ingest:",
                "      source: records",
                "      field: sequence",
                "    feature_bundle:",
                "      intermediate_block: 26",
                "      collect_log_likelihood: false",
                "      collect_output_layer_mean: false",
                "      collect_intermediate_embedding: true",
                "      sequence_view_inputs:",
                "        - dataset: usr_prom_eth_cip_anchor",
                f"          root: {usr_root}",
                "          view_selector:",
                "            product_kind: construct_insert",
                "          pooling:",
                "            operation: seq_mean",
                "    io:",
                "      write_back: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    events_path, policy = resolve_tool_events_path(tool="infer", config=config)

    assert events_path == (usr_root / "usr_prom_eth_cip_anchor" / ".events.log").resolve()
    assert policy == "infer"


def test_resolve_tool_events_path_infer_rejects_multi_dataset_sequence_view_config(tmp_path: Path) -> None:
    config = tmp_path / "infer_sequence_views.yaml"
    usr_root = tmp_path / "usr_root"
    config.write_text(
        "\n".join(
            [
                "model:",
                "  id: evo2_7b",
                "  device: cpu",
                "  precision: fp32",
                "  alphabet: dna",
                "jobs:",
                "  - id: main_sequence_views_7b",
                "    operation: extract",
                "    ingest:",
                "      source: records",
                "      field: sequence",
                "    feature_bundle:",
                "      intermediate_block: 26",
                "      collect_log_likelihood: false",
                "      collect_output_layer_mean: false",
                "      collect_intermediate_embedding: true",
                "      sequence_view_inputs:",
                "        - dataset: usr_prom_eth_cip_anchor",
                f"          root: {usr_root}",
                "          view_selector:",
                "            product_kind: construct_insert",
                "          pooling:",
                "            operation: seq_mean",
                "        - dataset: construct_prom_eth_cip_context",
                f"          root: {usr_root}",
                "          view_selector:",
                "            product_kind: realized_context",
                "          pooling:",
                "            operation: anchor_mean",
                "            bounds_from: sequence_view",
                "    io:",
                "      write_back: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(NotifyConfigError, match="multiple sequence-view event sources"):
        resolve_tool_events_path(tool="infer", config=config)


def test_resolve_tool_events_path_infer_requires_explicit_root_even_with_env(tmp_path: Path, monkeypatch) -> None:
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

    with pytest.raises(NotifyConfigError, match="requires ingest.root for source='usr' write-back jobs"):
        resolve_tool_events_path(tool="infer", config=config)


def test_resolve_tool_events_path_infer_requires_explicit_root_without_env(tmp_path: Path, monkeypatch) -> None:
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

    with pytest.raises(NotifyConfigError, match="requires ingest.root for source='usr' write-back jobs"):
        resolve_tool_events_path(tool="infer", config=config)


def test_resolve_tool_events_path_infer_rejects_ambiguous_destinations(tmp_path: Path) -> None:
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
        resolve_tool_events_path(tool="infer", config=config)


def test_resolve_tool_events_path_infer_requires_usr_writeback_job(tmp_path: Path) -> None:
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
        resolve_tool_events_path(tool="infer", config=config)


def test_resolve_tool_events_path_construct_from_explicit_output_root(tmp_path: Path) -> None:
    workspace = tmp_path / "construct_workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    config = workspace / "config.yaml"
    _write_construct_config(
        config,
        input_root="shared_inputs",
        output_root="outputs/usr_datasets",
    )

    events_path, policy = resolve_tool_events_path(tool="construct", config=config)

    assert (
        events_path == (workspace / "outputs" / "usr_datasets" / "construct" / "demo_window" / ".events.log").resolve()
    )
    assert policy == "construct"


def test_resolve_tool_events_path_construct_falls_back_to_input_root(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    usr_root = tmp_path / "shared_usr"
    _write_construct_config(
        config,
        input_root=usr_root,
        output_root=None,
    )

    events_path, policy = resolve_tool_events_path(tool="construct", config=config)

    assert events_path == (usr_root / "construct" / "demo_window" / ".events.log").resolve()
    assert policy == "construct"


def test_resolve_tool_events_path_construct_requires_job_output_dataset(tmp_path: Path) -> None:
    config = tmp_path / "construct.yaml"
    _write_construct_config(
        config,
        input_root="shared_inputs",
        output_root="outputs/usr_datasets",
        output_dataset=None,
    )

    with pytest.raises(NotifyConfigError, match="job.output.target.dataset"):
        resolve_tool_events_path(tool="construct", config=config)


def test_resolve_tool_events_path_construct_requires_input_root_contract(tmp_path: Path) -> None:
    config = tmp_path / "construct.yaml"
    _write_construct_config(
        config,
        input_root=None,
        output_root=None,
    )

    with pytest.raises(NotifyConfigError, match="job.input.source.root is required"):
        resolve_tool_events_path(tool="construct", config=config)


@pytest.mark.parametrize("tool_alias", ["infer_evo2", "infer-evo2"])
def test_resolve_tool_events_path_accepts_legacy_infer_alias(tool_alias: str, tmp_path: Path) -> None:
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

    events_path, policy = resolve_tool_events_path(tool=tool_alias, config=config)

    assert events_path == (usr_root / "infer_demo" / ".events.log").resolve()
    assert policy == "infer"


def test_resolve_tool_events_path_densegen_from_usr_output_config(tmp_path: Path) -> None:
    run_root = tmp_path / "workspace"
    run_root.mkdir(parents=True, exist_ok=True)
    config = run_root / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "densegen:",
                "  run:",
                f"    root: {run_root}",
                "    id: stress_ethanol_cipro",
                "  output:",
                "    targets: [usr]",
                "    usr:",
                "      dataset: densegen_prom_eth_cip_source",
                "      root: outputs/usr_datasets",
                "    schema:",
                "      bio_type: dna",
                "      alphabet: dna_4",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    events_path, policy = resolve_tool_events_path(tool="densegen", config=config)

    assert (
        events_path
        == (run_root / "outputs" / "usr_datasets" / "densegen_prom_eth_cip_source" / ".events.log").resolve()
    )
    assert policy == "densegen"


def test_resolve_tool_events_path_densegen_supports_shared_usr_root_outside_outputs(tmp_path: Path) -> None:
    run_root = tmp_path / "workspace"
    run_root.mkdir(parents=True, exist_ok=True)
    config = run_root / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "densegen:",
                "  run:",
                f"    root: {run_root}",
                "    id: stress_ethanol_cipro",
                "  output:",
                "    targets: [usr]",
                "    usr:",
                "      dataset: densegen_prom_eth_cip_source",
                f"      root: {tmp_path / 'external_usr'}",
                "    schema:",
                "      bio_type: dna",
                "      alphabet: dna_4",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    events_path, policy = resolve_tool_events_path(tool="densegen", config=config)

    assert events_path == (tmp_path / "external_usr" / "densegen_prom_eth_cip_source" / ".events.log").resolve()
    assert policy == "densegen"


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
    import dnadesign.notify.events.source as events_source_module

    parsed = ast.parse(inspect.getsource(events_source_module))
    imported_modules: set[str] = set()
    for node in ast.walk(parsed):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            imported_modules.add(str(node.module or ""))

    assert "yaml" not in imported_modules
    assert "os" not in imported_modules


def test_source_builtin_module_uses_public_tool_contracts_only() -> None:
    import dnadesign.notify.events.source_builtin as source_builtin_module

    parsed = ast.parse(inspect.getsource(source_builtin_module))
    imported_modules: set[str] = set()
    for node in ast.walk(parsed):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            imported_modules.add(str(node.module or ""))

    assert "dnadesign.construct" in imported_modules
    assert "dnadesign.densegen.contracts" in imported_modules
    assert "dnadesign.infer.contracts" in imported_modules
    assert not any(module.startswith("dnadesign.construct.src") for module in imported_modules)
    assert not any(module.startswith("dnadesign.densegen.src") for module in imported_modules)
    assert not any(module.startswith("dnadesign.infer.src") for module in imported_modules)
    assert not any(module.startswith("dnadesign._contracts") for module in imported_modules)
