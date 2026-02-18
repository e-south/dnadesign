"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/stage_a/test_background_pool_config.py

Config validation for background_pool inputs (schema 2.9).

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dnadesign.densegen.src.config import (
    BackgroundPoolMiningBudgetConfig,
    BackgroundPoolMiningConfig,
    BackgroundPoolSamplingConfig,
    load_config,
)


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
              path: outputs/tables/records.parquet
          generation:
            sequence_length: 20
            plan:
              - name: demo_plan
                quota: 1
                sampling:
                  include_inputs: [neutral_bg]
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


def test_background_pool_length_defaults_to_range_16_20() -> None:
    sampling = BackgroundPoolSamplingConfig(
        n_sites=8,
        mining=BackgroundPoolMiningConfig(
            batch_size=100,
            budget=BackgroundPoolMiningBudgetConfig(mode="fixed_candidates", candidates=2000),
        ),
    )
    assert sampling.length.policy == "range"
    assert sampling.length.range == (16, 20)
    assert sampling.length.exact is None


def test_background_pool_length_exact_allows_implicit_default_range() -> None:
    sampling = BackgroundPoolSamplingConfig(
        n_sites=8,
        mining=BackgroundPoolMiningConfig(
            batch_size=100,
            budget=BackgroundPoolMiningBudgetConfig(mode="fixed_candidates", candidates=2000),
        ),
        length={"policy": "exact", "exact": 18},
    )
    assert sampling.length.policy == "exact"
    assert sampling.length.exact == 18
    assert sampling.length.range is None


def test_background_pool_length_exact_rejects_explicit_range() -> None:
    with pytest.raises(ValueError, match="background_pool.sampling.length.range is not allowed when policy=exact"):
        BackgroundPoolSamplingConfig(
            n_sites=8,
            mining=BackgroundPoolMiningConfig(
                batch_size=100,
                budget=BackgroundPoolMiningBudgetConfig(mode="fixed_candidates", candidates=2000),
            ),
            length={"policy": "exact", "exact": 18, "range": [16, 20]},
        )
