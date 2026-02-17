"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/cli/commands/demo_matrix.py

Runs canonical OPAL demo workflows in isolated temp directories for pressure
testing and CI-style smoke validation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import contextlib
import io
import shutil
import tempfile
from pathlib import Path
from typing import Any

import typer

from ...core.utils import ExitCodes, OpalError, print_stdout
from ...reporting.verify_outputs import compare_selection_to_ledger, read_selection_table
from ...storage.ledger import LedgerReader
from ...storage.workspace import CampaignWorkspace
from ..registry import cli_command
from ._common import internal_error, json_out, load_cli_config, opal_error
from .campaign_reset import cmd_campaign_reset
from .ingest_y import cmd_ingest_y
from .init import cmd_init
from .run import cmd_run
from .validate import cmd_validate

DEMO_FLOWS = (
    "demo_rf_sfxi_topn",
    "demo_gp_topn",
    "demo_gp_ei",
)


def _run_cli_quiet(fn, /, **kwargs) -> dict[str, str]:
    out = io.StringIO()
    err = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        fn(**kwargs)
    return {"stdout": out.getvalue(), "stderr": err.getvalue()}


def _parse_rounds(raw: str) -> list[int]:
    txt = str(raw or "").strip()
    if not txt:
        raise OpalError("--rounds cannot be empty; expected comma-separated non-negative integers (e.g. 0 or 0,1).")
    out: list[int] = []
    for part in txt.split(","):
        token = part.strip()
        if not token:
            continue
        if not token.isdigit():
            raise OpalError(f"Invalid round token {token!r}; expected non-negative integers.")
        out.append(int(token))
    if not out:
        raise OpalError("--rounds did not contain any valid round values.")
    return out


def _campaigns_root() -> Path:
    return Path(__file__).resolve().parents[3] / "campaigns"


def _verify_round(cfg_path: Path, round_index: int) -> dict[str, Any]:
    cfg = load_cli_config(cfg_path)
    ws = CampaignWorkspace.from_config(cfg, cfg_path)
    reader = LedgerReader(ws)
    runs_df = reader.read_runs(columns=["run_id", "as_of_round"])
    rows = runs_df.loc[runs_df["as_of_round"].astype(int) == int(round_index)].copy()
    if rows.empty:
        raise OpalError(f"No run metadata found for round {int(round_index)}.")
    rows["run_id"] = rows["run_id"].astype(str)
    run_id = str(rows.sort_values(["run_id"]).tail(1).iloc[0]["run_id"])
    sel_path = ws.round_selection_dir(round_index) / "selection_top_k.csv"
    if not sel_path.exists():
        raise OpalError(f"Selection artifact missing: {sel_path}")
    selection_df = read_selection_table(sel_path)
    ledger_df = reader.read_predictions(
        columns=["id", "pred__score_selected"],
        round_selector=int(round_index),
        run_id=run_id,
    )
    summary, _ = compare_selection_to_ledger(selection_df, ledger_df, eps=1e-6)
    return {
        "round": int(round_index),
        "run_id": run_id,
        "rows_compared": int(summary["rows_compared"]),
        "mismatch_count": int(summary["mismatch_count"]),
        "max_abs_diff": float(summary["max_abs_diff"]),
        "selection_path": str(sel_path),
    }


def _run_demo_flow(*, flow_name: str, tmp_root: Path, rounds: list[int], fail_fast: bool) -> dict[str, Any]:
    campaigns_root = _campaigns_root()
    src = campaigns_root / flow_name
    if not src.exists():
        raise OpalError(f"Demo campaign not found: {src}")
    dst = tmp_root / flow_name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    base_records = campaigns_root / "demo" / "records.parquet"
    if not base_records.exists():
        raise OpalError(f"Base records missing: {base_records}")
    shutil.copy2(base_records, dst / "records.parquet")

    cfg_path = dst / "configs" / "campaign.yaml"
    labels_file = dst / "inputs" / "r0" / "vec8-b0.xlsx"
    if not labels_file.exists():
        raise OpalError(f"Demo labels file missing: {labels_file}")

    round_summaries: list[dict[str, Any]] = []
    try:
        _run_cli_quiet(cmd_campaign_reset, config=cfg_path, apply=True, backup=False, json=True)
        _run_cli_quiet(cmd_init, config=cfg_path, no_hints=True, json=True)
        _run_cli_quiet(cmd_validate, config=cfg_path, no_hints=True)
        for rr in rounds:
            _run_cli_quiet(
                cmd_ingest_y,
                config=cfg_path,
                round=int(rr),
                csv=labels_file,
                transform=None,
                params=None,
                apply=True,
                unknown_sequences="drop",
                infer_missing_required=False,
                if_exists="replace",
                no_hints=True,
                json=True,
            )
            _run_cli_quiet(
                cmd_run,
                config=cfg_path,
                round=int(rr),
                k=None,
                resume=bool(rr > 0),
                score_batch_size=None,
                verbose=False,
                no_hints=True,
                json=True,
            )
            round_summary = _verify_round(cfg_path, int(rr))
            round_summaries.append(round_summary)
            if int(round_summary["mismatch_count"]) > 0 and fail_fast:
                raise OpalError(
                    f"[{flow_name}] round {rr} has mismatches={round_summary['mismatch_count']} in verify check."
                )
    except Exception as e:
        if fail_fast:
            raise
        return {
            "flow": flow_name,
            "ok": False,
            "error": str(e),
            "rounds": round_summaries,
        }

    return {
        "flow": flow_name,
        "ok": all(int(r["mismatch_count"]) == 0 for r in round_summaries),
        "rounds": round_summaries,
    }


@cli_command(
    "demo-matrix",
    help="Run canonical demo workflows end-to-end in isolated temp copies.",
)
def cmd_demo_matrix(
    tmp_root: Path | None = typer.Option(None, "--tmp-root", help="Optional temp root for copied campaign runs."),
    rounds: str = typer.Option("0", "--rounds", help="Comma-separated rounds to execute (example: 0 or 0,1)."),
    keep: bool = typer.Option(False, "--keep", help="Keep temp directory after completion."),
    fail_fast: bool = typer.Option(False, "--fail-fast", help="Abort immediately on first flow failure."),
    json: bool = typer.Option(False, "--json/--human", help="Output format."),
) -> None:
    created_tmp = False
    try:
        parsed_rounds = _parse_rounds(rounds)
        if tmp_root is None:
            tmp_root = Path(tempfile.mkdtemp(prefix="opal-demo-matrix-"))
            created_tmp = True
        else:
            tmp_root = Path(tmp_root).resolve()
            tmp_root.mkdir(parents=True, exist_ok=True)

        flow_rows: list[dict[str, Any]] = []
        for flow_name in DEMO_FLOWS:
            flow_rows.append(
                _run_demo_flow(
                    flow_name=flow_name,
                    tmp_root=tmp_root,
                    rounds=parsed_rounds,
                    fail_fast=fail_fast,
                )
            )

        ok = all(bool(row.get("ok")) for row in flow_rows)
        summary = {
            "ok": ok,
            "tmp_root": str(tmp_root),
            "rounds": parsed_rounds,
            "flows": flow_rows,
        }

        if json:
            json_out(summary)
        else:
            print_stdout("demo-matrix")
            print_stdout(f"- tmp_root: {summary['tmp_root']}")
            print_stdout(f"- rounds: {summary['rounds']}")
            for row in flow_rows:
                print_stdout(f"- {row['flow']}: ok={row['ok']}")
                for rr in row.get("rounds", []):
                    print_stdout(
                        "  "
                        f"round={rr['round']} run_id={rr['run_id']} "
                        f"mismatches={rr['mismatch_count']} max_abs_diff={rr['max_abs_diff']}"
                    )
                if row.get("error"):
                    print_stdout(f"  error={row['error']}")
            print_stdout(f"- overall_ok: {ok}")

        if not ok:
            raise typer.Exit(code=ExitCodes.BAD_ARGS)
    except OpalError as e:
        opal_error("demo-matrix", e)
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        internal_error("demo-matrix", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)
    finally:
        if created_tmp and not keep and tmp_root is not None:
            try:
                shutil.rmtree(tmp_root)
            except Exception:
                pass
