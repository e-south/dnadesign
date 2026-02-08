"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/config_fixtures.py

Config writer helpers for DenseGen tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import textwrap
from pathlib import Path


def write_minimal_config(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            """
            densegen:
              schema_version: "2.9"
              run:
                id: demo
                root: "."
              inputs:
                - name: demo_input
                  type: binding_sites
                  path: inputs.csv
              output:
                targets: [parquet]
                schema:
                  bio_type: dna
                  alphabet: dna_4
                parquet:
                  path: outputs/tables/dense_arrays.parquet
              generation:
                sequence_length: 10
                plan:
                  - name: demo_plan
                    quota: 1
                    sampling:
                      include_inputs: [demo_input]
                    regulator_constraints:
                      groups: []
              solver:
                backend: CBC
                strategy: iterate
              logging:
                log_dir: outputs/logs
            """
        ).strip()
        + "\n"
    )
