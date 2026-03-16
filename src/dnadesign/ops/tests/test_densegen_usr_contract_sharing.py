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


def test_notify_and_ops_reference_shared_densegen_usr_output_contract_parser() -> None:
    import dnadesign.notify.events.source_builtin as notify_source_module
    import dnadesign.ops.orchestrator.usr_overlay_inputs as ops_inputs_module

    notify_source_text = inspect.getsource(notify_source_module)
    ops_inputs_text = inspect.getsource(ops_inputs_module)

    assert "resolve_densegen_usr_output_contract" in notify_source_text
    assert "resolve_densegen_usr_output_contract" in ops_inputs_text


def test_notify_and_ops_reference_shared_infer_usr_output_contract_parser() -> None:
    import dnadesign.notify.events.source_builtin as notify_source_module
    import dnadesign.ops.orchestrator.usr_overlay_inputs as ops_inputs_module

    notify_source_text = inspect.getsource(notify_source_module)
    ops_inputs_text = inspect.getsource(ops_inputs_module)

    assert "resolve_infer_usr_output_contract" in notify_source_text
    assert "resolve_infer_usr_output_contract" in ops_inputs_text


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
