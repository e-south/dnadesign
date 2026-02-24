"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/portfolio/test_portfolio_spec_strict_validation.py

Validate strict Portfolio spec contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dnadesign.cruncher.portfolio.schema_models import PortfolioRoot


def _base_payload() -> dict:
    return {
        "portfolio": {
            "schema_version": 3,
            "name": "pairwise_handoff",
            "execution": {"mode": "aggregate_only"},
            "artifacts": {"table_format": "parquet", "write_csv": True},
            "sources": [
                {
                    "id": "pairwise_cpxr_baer",
                    "workspace": "../pairwise_cpxr_baer",
                    "run_dir": "outputs/set1_cpxr_baer",
                }
            ],
        }
    }


def _base_payload_v2() -> dict:
    return {
        "portfolio": {
            "schema_version": 3,
            "name": "pairwise_handoff",
            "execution": {"mode": "prepare_then_aggregate"},
            "studies": {
                "ensure_specs": [
                    "configs/studies/length_vs_score.study.yaml",
                    "configs/studies/diversity_vs_score.study.yaml",
                ],
                "sequence_length_table": {
                    "enabled": True,
                    "study_spec": "configs/studies/length_vs_score.study.yaml",
                    "top_n_lengths": 6,
                },
            },
            "artifacts": {"table_format": "parquet", "write_csv": True},
            "sources": [
                {
                    "id": "pairwise_cpxr_baer",
                    "workspace": "../pairwise_cpxr_baer",
                    "run_dir": "outputs/set1_cpxr_baer",
                    "prepare": {
                        "runbook": "configs/runbook.yaml",
                        "step_ids": ["analyze_summary", "export_sequences"],
                    },
                    "study_spec": "configs/studies/diversity_vs_score.study.yaml",
                }
            ],
        }
    }


def test_spec_rejects_unknown_top_level_key() -> None:
    payload = _base_payload()
    payload["portfolio"]["extra_key"] = {"x": 1}
    with pytest.raises(ValidationError):
        PortfolioRoot.model_validate(payload)


def test_spec_rejects_duplicate_source_ids() -> None:
    payload = _base_payload()
    payload["portfolio"]["sources"] = [
        {
            "id": "pairwise_cpxr_baer",
            "workspace": "../pairwise_cpxr_baer",
            "run_dir": "outputs/set1_cpxr_baer",
        },
        {
            "id": "pairwise_cpxr_baer",
            "workspace": "../pairwise_cpxr_lexa",
            "run_dir": "outputs/set1_cpxr_lexa",
        },
    ]
    with pytest.raises(ValidationError, match="source ids must be unique"):
        PortfolioRoot.model_validate(payload)


def test_spec_rejects_invalid_source_id() -> None:
    payload = _base_payload()
    payload["portfolio"]["sources"][0]["id"] = "bad id"
    with pytest.raises(ValidationError, match="slug-safe"):
        PortfolioRoot.model_validate(payload)


def test_spec_rejects_legacy_top_k_key() -> None:
    payload = _base_payload()
    payload["portfolio"]["sources"][0]["top_k"] = 8
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        PortfolioRoot.model_validate(payload)


def test_spec_rejects_legacy_output_root_key() -> None:
    payload = _base_payload()
    payload["portfolio"]["artifacts"]["output_root"] = "outputs/portfolio_runs"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        PortfolioRoot.model_validate(payload)


def test_schema_v3_prepare_then_aggregate_requires_source_prepare_blocks() -> None:
    payload = _base_payload_v2()
    payload["portfolio"]["sources"][0].pop("prepare")
    with pytest.raises(ValidationError, match="requires prepare for every source"):
        PortfolioRoot.model_validate(payload)


def test_schema_v3_rejects_empty_prepare_step_ids() -> None:
    payload = _base_payload_v2()
    payload["portfolio"]["sources"][0]["prepare"]["step_ids"] = []
    with pytest.raises(ValidationError, match="step_ids must be non-empty"):
        PortfolioRoot.model_validate(payload)


def test_schema_v3_rejects_empty_study_spec() -> None:
    payload = _base_payload_v2()
    payload["portfolio"]["sources"][0]["study_spec"] = "   "
    with pytest.raises(ValidationError, match="study_spec must be non-empty"):
        PortfolioRoot.model_validate(payload)


def test_schema_v3_accepts_global_sequence_length_table_options() -> None:
    payload = _base_payload_v2()
    model = PortfolioRoot.model_validate(payload)
    assert model.portfolio.studies.sequence_length_table.enabled is True
    assert model.portfolio.studies.sequence_length_table.top_n_lengths == 6


def test_schema_v3_defaults_portfolio_tables_to_parquet_without_csv_mirror() -> None:
    payload = _base_payload()
    payload["portfolio"].pop("artifacts")
    model = PortfolioRoot.model_validate(payload)
    assert model.portfolio.artifacts.table_format == "parquet"
    assert model.portfolio.artifacts.write_csv is False


def test_schema_v3_defaults_max_parallel_sources_to_four() -> None:
    payload = _base_payload()
    model = PortfolioRoot.model_validate(payload)
    assert model.portfolio.execution.max_parallel_sources == 4


def test_schema_v3_defaults_portfolio_elite_showcase_plot_contract() -> None:
    payload = _base_payload()
    model = PortfolioRoot.model_validate(payload)
    assert model.portfolio.plots.elite_showcase.enabled is True
    assert model.portfolio.plots.elite_showcase.top_n_per_source is None
    assert model.portfolio.plots.elite_showcase.ncols == 5
    assert model.portfolio.plots.elite_showcase.plot_format == "pdf"


def test_schema_v3_rejects_selector_with_ids_and_ranks() -> None:
    payload = _base_payload()
    payload["portfolio"]["plots"] = {
        "elite_showcase": {
            "source_selectors": {
                "pairwise_cpxr_baer": {
                    "elite_ids": ["pairwise_cpxr_baer_elite_1"],
                    "elite_ranks": [1],
                }
            }
        }
    }
    with pytest.raises(ValidationError, match="exactly one of elite_ids or elite_ranks"):
        PortfolioRoot.model_validate(payload)


def test_schema_v3_rejects_selector_for_unknown_source() -> None:
    payload = _base_payload()
    payload["portfolio"]["plots"] = {
        "elite_showcase": {
            "source_selectors": {
                "missing_source": {
                    "elite_ranks": [1],
                }
            }
        }
    }
    with pytest.raises(ValidationError, match="source_selectors contains unknown source ids"):
        PortfolioRoot.model_validate(payload)


def test_schema_v3_rejects_invalid_max_parallel_sources() -> None:
    payload = _base_payload()
    payload["portfolio"]["execution"]["max_parallel_sources"] = 0
    with pytest.raises(ValidationError, match="max_parallel_sources must be >= 1"):
        PortfolioRoot.model_validate(payload)


def test_schema_v3_rejects_sequence_length_table_with_invalid_top_n() -> None:
    payload = _base_payload_v2()
    payload["portfolio"]["studies"]["sequence_length_table"]["top_n_lengths"] = 0
    with pytest.raises(ValidationError, match="top_n_lengths must be >= 1"):
        PortfolioRoot.model_validate(payload)


def test_schema_v3_requires_sequence_length_table_study_to_be_ensured() -> None:
    payload = _base_payload_v2()
    payload["portfolio"]["studies"]["ensure_specs"] = [
        "configs/studies/diversity_vs_score.study.yaml",
    ]
    with pytest.raises(ValidationError, match="must be listed in portfolio.studies.ensure_specs"):
        PortfolioRoot.model_validate(payload)


def test_portfolio_schema_rejects_legacy_schema_v2() -> None:
    payload = {
        "portfolio": {
            "schema_version": 2,
            "name": "legacy_handoff",
            "sources": [
                {
                    "id": "pairwise_cpxr_baer",
                    "workspace": "../pairwise_cpxr_baer",
                    "run_dir": "outputs/set1_cpxr_baer",
                }
            ],
        }
    }
    with pytest.raises(ValidationError, match="Portfolio schema v3 required"):
        PortfolioRoot.model_validate(payload)
