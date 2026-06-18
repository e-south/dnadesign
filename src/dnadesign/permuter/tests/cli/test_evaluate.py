"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/permuter/tests/cli/test_evaluate.py

CLI evaluation replacement and metric-materialization contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from dnadesign.permuter.src.cli.app import app


def test_evaluate_explicit_with_replaces_workspace_metrics(tmp_path: Path) -> None:
    workspace = tmp_path / "toy"
    outputs = workspace / "outputs"
    outputs.mkdir(parents=True)
    workspace.joinpath("refs.csv").write_text("ref_name,sequence\ntoy,ACGT\n", encoding="utf-8")
    workspace.joinpath("config.yaml").write_text(
        """
scope:
  name: toy
  bio_type: dna
  input:
    refs: "${WORKSPACE_DIR}/refs.csv"
    name_col: ref_name
    seq_col: sequence
  permute:
    protocol: scan_dna
    params: {}
  evaluate:
    metrics:
      - id: llr_mean
        evaluator: evo2_llr
        metric: log_likelihood_ratio
  output:
    dir: "${WORKSPACE_DIR}/outputs"
    layout: flat
""".strip()
        + "\n",
        encoding="utf-8",
    )
    sequence = "ACGT"
    pd.DataFrame(
        [
            {
                "id": hashlib.sha1(f"dna|{sequence}".encode("utf-8")).hexdigest(),
                "bio_type": "dna",
                "sequence": sequence,
                "alphabet": "dna_4",
                "length": len(sequence),
                "source": "unit",
                "created_at": "2026-05-24T00:00:00Z",
                "permuter__scope": "toy",
                "permuter__ref": "toy",
                "permuter__protocol": "scan_dna",
                "permuter__var_id": "toy-variant",
                "permuter__modifications": [],
                "permuter__round": 1,
            }
        ]
    ).to_parquet(outputs / "records.parquet", index=False)
    (outputs / "REF.fa").write_text(">toy\nACGT\n", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        ["evaluate", "--workspace", str(workspace), "--ref", "toy", "--with", "smoke:placeholder:log_likelihood"],
    )

    assert result.exit_code == 0, result.output
    df = pd.read_parquet(outputs / "records.parquet")
    assert "permuter__observed__smoke" in df.columns
    assert "permuter__observed__llr_mean" not in df.columns
