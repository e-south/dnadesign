"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/config/test_plan_include_inputs.py

Config validation for plan-scoped include_inputs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from dnadesign.densegen.src.config import load_config


def _write_config(tmp_path: Path, plan_block: str) -> Path:
    cfg_path = tmp_path / "config.yaml"
    plan_lines = textwrap.dedent(plan_block).strip("\n")
    indented_plan = textwrap.indent(plan_lines, " " * 6)
    base = textwrap.dedent(
        """
        densegen:
          schema_version: "2.9"
          run:
            id: demo
            root: "."
          inputs:
            - name: seqs
              type: sequence_library
              path: inputs/seqs.csv
          output:
            targets: [parquet]
            schema:
              bio_type: dna
              alphabet: dna_4
            parquet:
              path: outputs/tables/dense_arrays.parquet
          generation:
            sequence_length: 20
            sampling:
              pool_strategy: subsample
              library_size: 2
              library_sampling_strategy: tf_balanced
            plan:
        """
    ).strip()
    tail_lines = textwrap.dedent(
        """
          solver:
            backend: CBC
            strategy: iterate
          logging:
            log_dir: outputs/logs
        """
    ).strip()
    tail = textwrap.indent(tail_lines, " " * 2)
    cfg_path.write_text(base + "\n" + indented_plan + "\n" + tail + "\n")
    return cfg_path


def test_plan_include_inputs_required(tmp_path: Path) -> None:
    cfg_path = _write_config(
        tmp_path,
        """
          - name: demo_plan
            quota: 1
            sampling: {}
            regulator_constraints:
              groups: []
        """,
    )
    with pytest.raises(Exception, match="include_inputs"):
        load_config(cfg_path)


def test_plan_include_inputs_unknown_input(tmp_path: Path) -> None:
    cfg_path = _write_config(
        tmp_path,
        """
          - name: demo_plan
            quota: 1
            sampling:
              include_inputs: [missing]
            regulator_constraints:
              groups: []
        """,
    )
    with pytest.raises(Exception, match="include_inputs"):
        load_config(cfg_path)
