"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/cli/test_cli_demo_matrix.py

CLI tests for demo workflow matrix command wiring and summary output.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from typer.testing import CliRunner

from dnadesign.opal.src.cli.app import _build
from dnadesign.opal.src.cli.commands import demo_matrix as demo_matrix_cmd
from dnadesign.opal.src.core.utils import ExitCodes
from dnadesign.opal.src.storage.x_contracts import validate_x_parquet_column

DEMO_X_COLUMN = "infer__evo2_7b__60bp_dual_promoter_cpxR_LexA__logits_mean"


def test_demo_matrix_json_summary_shape(monkeypatch, tmp_path: Path) -> None:
    def _fake_run_demo_flow(*, flow_name: str, tmp_root: Path, rounds: list[int], fail_fast: bool) -> dict:
        _ = tmp_root
        _ = fail_fast
        return {
            "flow": flow_name,
            "ok": True,
            "rounds": [{"round": r, "mismatch_count": 0} for r in rounds],
        }

    monkeypatch.setattr(demo_matrix_cmd, "_run_demo_flow", _fake_run_demo_flow)
    app = _build()
    runner = CliRunner()
    res = runner.invoke(
        app,
        ["--no-color", "demo-matrix", "--tmp-root", str(tmp_path), "--rounds", "0,1", "--json"],
    )
    assert res.exit_code == 0, res.stdout
    out = json.loads(res.stdout)
    assert out["ok"] is True
    assert out["rounds"] == [0, 1]
    assert len(out["flows"]) == len(demo_matrix_cmd.DEMO_FLOWS)


def test_run_cli_quiet_treats_zero_typer_exit_as_success() -> None:
    import typer

    def _exits_zero() -> None:
        raise typer.Exit(code=0)

    captured = demo_matrix_cmd._run_cli_quiet(_exits_zero)

    assert captured == {"stdout": "", "stderr": ""}


def test_demo_matrix_base_records_x_contract_is_canonical() -> None:
    records_path = demo_matrix_cmd._campaigns_root() / "demo" / "records.parquet"
    report = validate_x_parquet_column(records_path, x_column=DEMO_X_COLUMN)
    x_type = pq.ParquetFile(records_path).schema_arrow.field(DEMO_X_COLUMN).type

    assert pa.types.is_fixed_size_list(x_type)
    assert x_type.list_size == 512
    assert report.row_count > 0
    assert report.x_dim == 512
    assert report.value_type == "double"


def test_demo_matrix_json_failure_exits_bad_args_without_internal_error(monkeypatch, tmp_path: Path) -> None:
    def _fake_run_demo_flow(*, flow_name: str, tmp_root: Path, rounds: list[int], fail_fast: bool) -> dict:
        _ = tmp_root
        _ = rounds
        _ = fail_fast
        return {
            "flow": flow_name,
            "ok": False,
            "error": "contract failure",
            "rounds": [],
        }

    monkeypatch.setattr(demo_matrix_cmd, "_run_demo_flow", _fake_run_demo_flow)
    app = _build()
    runner = CliRunner()
    res = runner.invoke(
        app,
        ["--no-color", "demo-matrix", "--tmp-root", str(tmp_path), "--rounds", "0", "--json"],
    )

    assert res.exit_code == ExitCodes.BAD_ARGS, res.stdout
    out = json.loads(res.stdout)
    assert out["ok"] is False
    assert {row["error"] for row in out["flows"]} == {"contract failure"}
    assert "Internal error" not in res.stdout
    assert "Internal error" not in res.stderr
