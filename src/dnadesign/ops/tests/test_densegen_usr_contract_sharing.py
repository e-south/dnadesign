"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/tests/test_densegen_usr_contract_sharing.py

Architecture contract tests for owner-owned USR contract parsing across tools.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import dnadesign.usr as usr_roots
from dnadesign.construct.contracts import resolve_construct_usr_output_contract
from dnadesign.densegen.contracts import resolve_densegen_usr_output_contract
from dnadesign.infer.contracts import resolve_infer_usr_output_contract
from dnadesign.notify.events.source_builtin import (
    _resolve_construct_events_from_config,
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
              inputs:
                - name: library
                  type: sequence_library
                  path: outputs/inputs/library.fa
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
                round_robin: false
                max_accepted_per_library: 2
              solver:
                strategy: approximate
              logging:
                log_dir: outputs/logs
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    library_path = config_path.parent / "outputs" / "inputs" / "library.fa"
    library_path.parent.mkdir(parents=True, exist_ok=True)
    library_path.write_text(">seq1\nACGTACGTACGT\n", encoding="utf-8")


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


def _write_construct_config(
    config_path: Path,
    *,
    input_root: str | Path,
    output_root: str | Path,
) -> None:
    lines = [
        "job:",
        "  id: slot_a_window",
        "  input:",
        "    source:",
        "      kind: usr",
        "      dataset: anchors_demo",
        f"      root: {Path(input_root).as_posix() if isinstance(input_root, Path) else input_root}",
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
        "      dataset: construct/demo_window",
        f"      root: {Path(output_root).as_posix() if isinstance(output_root, Path) else output_root}",
    ]
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_densegen_shared_usr_contract_supports_notify_and_ops(tmp_path: Path) -> None:
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


def test_infer_shared_usr_contract_supports_notify_and_ops(tmp_path: Path) -> None:
    config_path = tmp_path / "infer_workspace" / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    _write_infer_write_back_config(config_path)

    contract = resolve_infer_usr_output_contract(config_path)
    overlay_inputs = parse_usr_overlay_guard_inputs(tool="infer", config_path=config_path)
    events_path = _resolve_infer_events_from_config(config_path)

    assert overlay_inputs.usr_root == contract.usr_root
    assert overlay_inputs.usr_dataset == contract.usr_dataset
    assert events_path == (contract.usr_root / contract.usr_dataset / ".events.log").resolve()


def test_construct_shared_usr_contract_supports_notify_and_ops(tmp_path: Path) -> None:
    config_path = tmp_path / "construct_workspace" / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    _write_construct_config(
        config_path,
        input_root="shared_inputs",
        output_root="outputs/usr_datasets",
    )

    contract = resolve_construct_usr_output_contract(config_path)
    overlay_inputs = parse_usr_overlay_guard_inputs(tool="construct", config_path=config_path)
    events_path = _resolve_construct_events_from_config(config_path)

    assert overlay_inputs.usr_root == contract.usr_root
    assert overlay_inputs.usr_dataset == contract.usr_dataset
    assert events_path == (contract.usr_root / contract.usr_dataset / ".events.log").resolve()


def test_usr_root_helpers_are_usr_owned() -> None:
    package_path = Path(usr_roots.__file__).resolve()
    assert package_path.as_posix().endswith("/src/dnadesign/usr/__init__.py")
    assert usr_roots.pkg_usr_root() == package_path.parent.resolve()
