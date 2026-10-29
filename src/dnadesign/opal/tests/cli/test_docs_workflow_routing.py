"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/cli/test_docs_workflow_routing.py

Documentation routing contracts for OPAL cross-tool workflow ownership.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from dnadesign.opal.src.config.loader import load_config
from dnadesign.opal.src.transforms_y.scalar_from_table_v1 import scalar_from_table_v1


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


def _read(rel_path: str) -> str:
    return (_repo_root() / rel_path).read_text(encoding="utf-8")


def test_opal_docs_index_routes_to_usr_infer_x_workflow() -> None:
    docs_index = _read("src/dnadesign/opal/docs/index.md")
    top_readme = _read("src/dnadesign/opal/README.md")

    assert "./workflows/seqledger-infer-x-active-learning.md" in docs_index
    assert "docs/index.md" in top_readme
    assert "docs/workflows/seqledger-infer-x-active-learning.md" in top_readme
    assert "../../../cluster/docs/workflows/exploratory-clustering.md" in _read(
        "src/dnadesign/opal/docs/workflows/seqledger-infer-x-active-learning.md"
    )


def test_opal_usr_infer_x_workflow_keeps_upstream_preconditions_explicit() -> None:
    workflow = _read("src/dnadesign/opal/docs/workflows/seqledger-infer-x-active-learning.md")

    assert "**Type:** workflow" in workflow
    assert "**Plane:** downstream-tool" in workflow
    assert "**Owner-boundary:** opal" in workflow
    assert "**Registry-id:** opal.downstream.seqledger-infer-x-active-learning" in workflow
    assert "**Execution-kind:** round-loop" in workflow
    assert "**Status-kind:** opal-campaign-state" in workflow
    assert "data.location.kind: usr" in workflow
    assert "x_column_name" in workflow
    assert "../../../seqledger/docs/operations/promoter/characterization-feature-matrix.md" in workflow


def test_usr_workflow_yaml_loads_and_maps_its_measurement_column(tmp_path: Path) -> None:
    workflow = _read("src/dnadesign/opal/docs/workflows/seqledger-infer-x-active-learning.md")
    blocks = re.findall(r"```yaml\n(.*?)```", workflow, flags=re.DOTALL)
    config_path = tmp_path / "campaign.yaml"
    config_path.write_text("\n".join(blocks), encoding="utf-8")
    config = load_config(config_path)
    labels = pd.DataFrame({"sequence": ["ACGT"], "measured_activity": [1.25]})
    transformed = scalar_from_table_v1(labels, config.data.transforms_y.params)
    assert transformed["y"].tolist() == [[1.25]]


def test_opal_cli_reference_documents_the_explicit_usr_coordinate() -> None:
    cli = _read("src/dnadesign/opal/docs/reference/cli.md")

    assert "opal --seqledger-root <absolute-path> <command>" in cli


def test_opal_cli_reference_documents_one_history_import_contract() -> None:
    cli = _read("src/dnadesign/opal/docs/reference/cli.md")

    assert "opal history import" in cli


def test_opal_notebook_reference_documents_all_round_review_scope() -> None:
    cli = _read("src/dnadesign/opal/docs/reference/cli.md")

    assert "[--round <latest|all|k>]" in cli
