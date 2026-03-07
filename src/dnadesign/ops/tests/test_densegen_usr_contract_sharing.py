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
