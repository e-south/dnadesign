"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/test_background_pool_config.py

Config validation for background_pool inputs (schema 2.9).

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dnadesign.densegen.src.config import load_config


def test_background_pool_requires_schema_29(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """
        densegen:
          schema_version: "2.8"
          run:
            id: demo
            root: "."
          inputs:
            - name: neutral_bg
              type: background_pool
              sampling:
                n_sites: 10
                mining:
                  batch_size: 10
                  budget:
                    mode: fixed_candidates
                    candidates: 100
                length:
                  policy: range
                  range: [16, 20]
                uniqueness:
                  key: sequence
          output:
            targets: [parquet]
            schema:
              bio_type: dna
              alphabet: dna_4
            parquet:
              path: outputs/tables/dense_arrays.parquet
          generation:
            sequence_length: 20
            quota: 1
            plan:
              - name: demo_plan
                quota: 1
                regulator_constraints:
                  groups: []
          solver:
            backend: CBC
            strategy: iterate
          logging:
            log_dir: outputs/logs
        """.strip()
        + "\n"
    )
    with pytest.raises(Exception, match="schema_version"):
        load_config(cfg_path)
