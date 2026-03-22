"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/tests/test_densegen_usr_contract_sharing.py

Architecture contract tests for shared DenseGen USR output parsing across tools.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import inspect
import textwrap
from pathlib import Path

from dnadesign._contracts import resolve_densegen_usr_output_contract, resolve_infer_usr_output_contract
from dnadesign.notify.events.source_builtin import (
    _resolve_densegen_events_from_config,
    _resolve_infer_events_from_config,
)
from dnadesign.ops.orchestrator.usr_overlay_inputs import parse_usr_overlay_guard_inputs


def _write_densegen_config(config_path: Path) -> None:
    config_path.write_text(
        textwrap.dedent(
            """
            densegen:
              schema_version: "2.9"
              run:
                id: demo
                root: .
              output:
                targets: [usr]
                schema:
                  bio_type: dna
                  alphabet: dna_4
                usr:
                  root: ../shared_usr
                  dataset: densegen/demo
                  chunk_size: 128
              generation:
                sequence_length: 12
                plan:
                  - name: default
                    sequences: 2
              runtime:
                max_accepted_per_library: 1
                round_robin: false
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )


def _write_infer_write_back_config(config_path: Path) -> None:
    config_path.write_text(
        textwrap.dedent(
            """
            model:
              id: evo2_7b
              device: cpu
              precision: fp32
              alphabet: dna
            jobs:
              - id: j1
                operation: extract
                ingest:
                  source: usr
                  dataset: demo
                  root: ../shared_usr
                  field: sequence
                outputs:
                  - id: ll
                    fn: evo2.log_likelihood
                    format: float
                io:
                  write_back: true
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )


def test_notify_and_ops_reference_shared_densegen_usr_output_contract_parser() -> None:
    import dnadesign.notify.events.source_builtin as notify_source_module
    import dnadesign.ops.orchestrator.usr_overlay_inputs as ops_inputs_module

    notify_source_text = inspect.getsource(notify_source_module)
    ops_inputs_text = inspect.getsource(ops_inputs_module)

    assert "resolve_densegen_usr_output_contract" in notify_source_text
    assert "resolve_densegen_usr_output_contract" in ops_inputs_text


def test_notify_and_ops_reference_shared_infer_usr_output_contract_parser(tmp_path: Path) -> None:
    config_path = tmp_path / "infer_workspace" / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    _write_infer_write_back_config(config_path)

    contract = resolve_infer_usr_output_contract(config_path)
    overlay_inputs = parse_usr_overlay_guard_inputs(tool="infer", config_path=config_path)
    events_path = _resolve_infer_events_from_config(config_path)

    assert overlay_inputs.usr_root == contract.usr_root
    assert overlay_inputs.usr_dataset == contract.usr_dataset
    assert events_path == (contract.usr_root / contract.usr_dataset / ".events.log").resolve()


def test_densegen_shared_usr_root_contract_supports_notify_and_ops(tmp_path: Path) -> None:
    config_path = tmp_path / "densegen_workspace" / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    _write_densegen_config(config_path)

    contract = resolve_densegen_usr_output_contract(config_path)
    overlay_inputs = parse_usr_overlay_guard_inputs(tool="densegen", config_path=config_path)
    events_path = _resolve_densegen_events_from_config(config_path)

    assert contract.usr_root == (tmp_path / "shared_usr").resolve()
    assert overlay_inputs.usr_root == contract.usr_root
    assert overlay_inputs.usr_dataset == contract.usr_dataset
    assert events_path == (contract.usr_root / contract.usr_dataset / ".events.log").resolve()


def test_notify_and_ops_reference_shared_construct_usr_output_contract_parser() -> None:
    import dnadesign.notify.events.source_builtin as notify_source_module
    import dnadesign.ops.orchestrator.usr_overlay_inputs as ops_inputs_module

    notify_source_text = inspect.getsource(notify_source_module)
    ops_inputs_text = inspect.getsource(ops_inputs_module)

    assert "resolve_construct_usr_output_contract" in notify_source_text
    assert "resolve_construct_usr_output_contract" in ops_inputs_text


def test_construct_and_infer_contracts_reference_shared_usr_root_resolution_helpers() -> None:
    import dnadesign._contracts.construct_usr_output as construct_contract_module
    import dnadesign._contracts.usr_producer as usr_producer_module
    import dnadesign.infer.src.ingest.sources as infer_sources_module

    construct_text = inspect.getsource(construct_contract_module)
    producer_text = inspect.getsource(usr_producer_module)
    infer_sources_text = inspect.getsource(infer_sources_module)

    assert "from dnadesign.usr_roots import" in construct_text
    assert "from dnadesign.usr_roots import" in producer_text
    assert "from dnadesign.usr_roots import" in infer_sources_text
